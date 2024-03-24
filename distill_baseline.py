import datetime
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm, trange
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, TensorDataset, epoch, get_loops, match_loss, ParamDiffAug, Conv3DNet
import wandb
import copy
import random
from reparam_module import ReparamModule
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import distill_utils

def main(args):
    if args.outer_loop is None and args.inner_loop is None:
        args.outer_loop, args.inner_loop = get_loops(args.ipc)
    elif args.outer_loop is None or args.inner_loop is None:
        raise ValueError(f"Please set neither or both outer/inner_loop: {args.outer_loop}, {args.inner_loop}")
    print('outer_loop = %d, inner_loop = %d'%(args.outer_loop, args.inner_loop))

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    print('Evaluation iterations: ', eval_it_pool)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader= get_dataset(args.dataset, args.data_path)
    if args.preload:
        print("Preloading dataset")
        video_all = []
        label_all = []
        for i in trange(len(dst_train)):
            _ = dst_train[i]
            video_all.append(_[0])
            label_all.append(_[1])
        video_all = torch.stack(video_all)
        label_all = torch.tensor(label_all)
        dst_train = torch.utils.data.TensorDataset(video_all, label_all)
    
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    project_name = "Baseline_{}".format(args.method)

    wandb.init(sync_tensorboard=False,
               project=project_name,
               job_type="CleanRepo",
               config=args,
               name = f'{args.dataset}_ipc{args.ipc}_{args.lr_img}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
               )
    
    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    labels_all = label_all if args.preload else dst_train.labels
    indices_class = [[] for c in range(num_classes)]

    print("BUILDING DATASET")
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        if n == 1:
            imgs = dst_train[idx_shuffle[0]][0].unsqueeze(0)
        else:
            imgs = torch.cat([dst_train[i][0].unsqueeze(0) for i in idx_shuffle], 0)
        return imgs.to(args.device)

    image_syn = torch.randn(size=(num_classes*args.ipc, args.frames, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor(np.stack([np.ones(args.ipc)*i for i in range(0, num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    syn_lr = torch.tensor(args.lr_teacher).to(args.device) if args.method == 'MTT' else None

    if args.init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(0, num_classes):
            i = c 
            image_syn.data[i*args.ipc:(i+1)*args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')

    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(args.train_lr) if args.method == 'MTT' else None
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) # optimizer_img for synthetic data
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5) if args.train_lr else None
    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}
    
    if args.method == "MTT":
        if args.train_lr: 
            print("Train synthetic lr")
            optimizer_lr.zero_grad()

        expert_files = []
        n = 0
        while os.path.exists(os.path.join(args.buffer_path, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(args.buffer_path, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(args.buffer_path))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)

        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        random.shuffle(buffer)

        for it in trange(0, args.Iteration+1, ncols=60):
            ''' Evaluate synthetic data '''
            if it % 1000 == 0 :
                image_save = image_syn.detach()
                save_dir = os.path.join(".", "logged_files", project_name, wandb.run.name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
             
            if it in eval_it_pool:
                
                save_this_best_ckpt = False
                for model_eval in model_eval_pool:
                    print('Evaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    accs_test = []
                    accs_train = []
                    accs_per = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                        image_syn_eval, label_syn_eval = image_syn.detach().clone(), label_syn.detach().clone() # avoid any unaware modification
                        args.lr_net = syn_lr.detach()
                        _, acc_train, acc_test, acc_per = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, mode='none',test_freq=200)

                        accs_test.append(acc_test)
                        accs_train.append(acc_train)
                        accs_per.append(acc_per)
                    accs_test = np.array(accs_test)
                    accs_train = np.array(accs_train)
                    accs_per = np.array(accs_per)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    acc_per_mean = np.mean(accs_per, axis=0)
                    print("acc_per_mean:",acc_per_mean)
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_best_ckpt = True
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                        len(accs_test), model_eval, acc_test_mean, acc_test_std))
                    wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                    wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                    wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                    wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

            if it in eval_it_pool and (save_this_best_ckpt or it % 1000 == 0):
                image_save = image_syn.detach()
                save_dir = os.path.join(args.save_path, project_name, wandb.run.name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                if save_this_best_ckpt:
                    save_this_best_ckpt = False
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))
  
            wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)
            student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

            student_net = ReparamModule(student_net)

            if args.distributed:
                student_net = torch.nn.DataParallel(student_net)

            student_net.train()

            num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                random.shuffle(buffer)

            start_epoch = np.random.randint(0, args.max_start_epoch)
            starting_params = expert_trajectory[start_epoch]

            target_params = expert_trajectory[start_epoch+args.expert_epochs]
            target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

            student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

            syn_images = image_syn

            y_hat = label_syn.to(args.device)

            param_loss_list = []
            param_dist_list = []
            indices_chunks = []

            for step in range(args.syn_steps):

                if not indices_chunks:
                    indices = torch.randperm(len(syn_images))
                    indices_chunks = list(torch.split(indices, args.batch_syn))

                these_indices = indices_chunks.pop()


                x = syn_images[these_indices]
                this_y = y_hat[these_indices]

                if args.distributed:
                    forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    forward_params = student_params[-1]
                x = student_net(x, flat_param=forward_params)
                ce_loss = criterion(x, this_y)

                grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]

                student_params.append(student_params[-1] - syn_lr * grad)


            param_loss = torch.tensor(0.0).to(args.device)
            param_dist = torch.tensor(0.0).to(args.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            #print("param_dist: ", param_dist.item())

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)


            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            grand_loss = param_loss

            optimizer_img.zero_grad()
            if args.train_lr: 
                optimizer_lr.zero_grad()

            grand_loss.backward()

            optimizer_img.step()
            if args.train_lr: 
                optimizer_lr.step()
                syn_lr.data = syn_lr.data.clip(min=0.001)

            wandb.log({"Grand_Loss/{}".format(start_epoch): grand_loss.detach().cpu(),
            "Grand_Loss": grand_loss.detach().cpu(),
            "Start_Epoch": start_epoch})

            for _ in student_params:
                del _

    elif args.method == "DM":
        for it in trange(0, args.Iteration+1, ncols=60):
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                save_this_best_ckpt = False
                for model_eval in model_eval_pool:
                    print('Evaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                    accs_test = []
                    accs_train = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                        image_syn_eval, label_syn_eval = image_syn.detach().clone(), label_syn.detach().clone() # avoid any unaware modification
                        _, acc_train, acc_test, acc_per_cls = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, mode='none',test_freq=100)

                        accs_test.append(acc_test)
                        accs_train.append(acc_train)
                        print("acc_per_cls:",acc_per_cls)
                    accs_test = np.array(accs_test)
                    accs_train = np.array(accs_train)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_best_ckpt = True
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                        len(accs_test), model_eval, acc_test_mean, acc_test_std))
                    wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                    wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                    wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                    wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)
            
            if it in eval_it_pool and (save_this_best_ckpt or it % 1000 == 0):
                image_save = image_syn.detach()
                save_dir = os.path.join(args.save_path, project_name, wandb.run.name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                if save_this_best_ckpt:
                    save_this_best_ckpt = False
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))

            net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if args.distributed else net.embed

            loss_avg = 0

            loss = torch.tensor(0.0).to(args.device)
            for c in range(0,num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, args.frames, channel, im_size[0], im_size[1]))

                output_real = embed(img_real).detach()
                output_syn = embed(img_syn)

                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            optimizer_img.zero_grad()
            loss.backward()
            optimizer_img.step()
            loss_avg += loss.item()

            loss_avg /= (num_classes)


            wandb.log({"Loss": loss_avg}, step=it)
   

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='dataset')

    parser.add_argument('--method', type=str, default='DC', help='MTT or DM')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='use top5 to eval top5 accuracy, use S to eval single accuracy')
    
    parser.add_argument('--outer_loop', type=int, default=None, help='')
    parser.add_argument('--inner_loop', type=int, default=None, help='')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=50, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=1000, help='how many distillation steps to perform')

    parser.add_argument('--lr_net', type=float, default=0.001, help='learning rate for network')
    parser.add_argument('--lr_img', type=float, default=1, help='learning rate for synthetic data')
    parser.add_argument('--lr_lr', type=float, default=1e-5, help='learning rate for synthetic data')
    parser.add_argument('--lr_teacher', type=float, default=0.001, help='learning rate for teacher')
    parser.add_argument('--train_lr', action='store_true', help='train synthetic lr')
    
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_syn', type=int, default=None, help='batch size for syn')

    parser.add_argument('--init', type=str, default='real', choices=['noise', 'real', 'real-all'], help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--data_path', type=str, default='distill_utils/data', help='dataset path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=64, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')

    parser.add_argument('--buffer_path', type=str, default=None, help='buffer path')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--preload', action='store_true', help='preload dataset')
    parser.add_argument('--save_path',type=str, default='./logged_files', help='path to save')
    parser.add_argument('--frames', type=int, default=16, help='')


    args = parser.parse_args()

    main(args)

