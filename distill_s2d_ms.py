import datetime
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm, trange
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, Conv3DNet, MultiStaticSharedDataset
import wandb
import copy
import random
from reparam_module import ReparamModule
from torchvision.utils import save_image
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import distill_utils

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(args.startIt, args.Iteration + 1, args.eval_it).tolist()
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

    data_save = []

    project_name = "S2D_multis_{}".format(args.method)

    wandb.init(sync_tensorboard=False,
               project=project_name,
               job_type="CleanRepo",
               config=args,
            name = f'{args.dataset}_ipc{args.vpc}_{args.lr_dynamic}_{args.lr_hal}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'
               )
    
    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    if args.batch_syn is None:
        args.batch_syn = num_classes *args.vpc 


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

    def get_images(c, n):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        if n == 1:
            imgs = dst_train[idx_shuffle[0]][0].unsqueeze(0)
        else:
            imgs = torch.cat([dst_train[i][0].unsqueeze(0) for i in idx_shuffle], 0)
        return imgs.to(args.device)

    static_syn = torch.randn(size=(num_classes*args.spc, 3, im_size[0], im_size[1]), dtype=torch.float) #默认spc=1
    dynamic_syn = torch.randn(size=(num_classes, args.dpc, args.frames, 1, im_size[0], im_size[1]), dtype=torch.float)

    ''' initialize the hallucinator '''
    hals = nn.ModuleList([Conv3DNet() for _ in range(args.n_hal)])
    syn_lr = torch.tensor(args.lr_teacher)

    if args.path_static is not None:
        static_syn = torch.load(args.path_static)["image"]
        print('load static memory from %s' % args.path_static)
        print('static_syn shape: ', static_syn.shape)

    static_syn = static_syn.detach().to(args.device).requires_grad_(False) if args.no_train_static else static_syn.detach().to(args.device).requires_grad_(True)
    dynamic_syn = dynamic_syn.detach().to(args.device).requires_grad_(True)
    hals = hals.to(args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(args.train_lr)
    optimizer_static = None if args.no_train_static else torch.optim.SGD([static_syn], lr=args.lr_static, momentum=0.95)
    optimizer_dynamic = torch.optim.SGD([dynamic_syn], lr=args.lr_dynamic, momentum=0.95)
    optimizer_hals = torch.optim.SGD(hals.parameters(), lr=args.lr_hal, momentum=0.95)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.9) if args.train_lr else None

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins' % get_time())

    if args.method == "MTT":
        expert_dir = args.buffer_path
        print("Expert Dir: {}".format(expert_dir))

        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        random.shuffle(buffer)

        best_acc = {m: 0 for m in model_eval_pool}
        best_std = {m: 0 for m in model_eval_pool}

        for it in range(0, args.Iteration + 1):
            save_this_it = False
            wandb.log({"Progress": it}, step=it)
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                    args.model, model_eval, it))

                    accs_test = []
                    accs_train = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                            args.device)  # get a random model
                        
                        static_syn_eval = copy.deepcopy(static_syn.detach())
                        dynamic_syn_eval = copy.deepcopy(dynamic_syn.detach())
                        hal_eval = copy.deepcopy(hals)

                        args.lr_net = syn_lr.detach()
                        _, acc_train, acc_test, acc_per_cls = evaluate_synset(it_eval, net_eval,
                                                                [static_syn_eval, dynamic_syn_eval, hal_eval],
                                                                None, testloader, args, mode='multi-static')
                        print("acc_per_cls: {}".format(acc_per_cls))
                        accs_test.append(acc_test)
                        accs_train.append(acc_train)
                    accs_test = np.array(accs_test)
                    accs_train = np.array(accs_train)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_it = True
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                    len(accs_test), model_eval, acc_test_mean, acc_test_std))
                    wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                    wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                    wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                    wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

            if it in eval_it_pool and (save_this_it or it % 1000 == 0):
                with torch.no_grad():
                    image_save = static_syn.detach()
                    dynamic_save = dynamic_syn.flatten(0, 1).detach()

                    save_dir = os.path.join(args.save_path, project_name, wandb.run.name)

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if not args.no_train_static:
                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                    torch.save(hals.state_dict(), os.path.join(save_dir, "hal_{}.pt".format(it)))
                    torch.save(dynamic_save.cpu(), os.path.join(save_dir, "dynamic_{}.pt".format(it)))

                    if save_this_it:
                        if not args.no_train_static:
                            torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))
                        torch.save(hals.state_dict(), os.path.join(save_dir, "weights_best.pt"))
                        torch.save(dynamic_save.cpu(), os.path.join(save_dir, "dynamic_best.pt"))

            wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

            student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(
                args.device)  # get a random model
            
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
                print("loading file {}".format(expert_files[file_idx]))
                random.shuffle(buffer)

            start_epoch = np.random.randint(0, args.max_start_epoch)
            starting_params = expert_trajectory[start_epoch]

            target_params = expert_trajectory[start_epoch + args.expert_epochs]
            target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

            student_params = [
                torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

            syn_static = static_syn
            syn_dynamic = dynamic_syn

            param_loss_list = []
            param_dist_list = []
            indices_chunks = []

            for step in range(args.syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(num_classes*args.vpc, device=args.device)
                    indices_chunks = list(torch.split(indices, args.batch_syn))
                these_indices = indices_chunks.pop() 
                label = these_indices // args.vpc
                idx = these_indices % args.vpc
                dynamic_idx =  2*idx + torch.randint(2, (these_indices.shape[0],), device=args.device)
                static_idx = args.spc * label + 2*idx + torch.randint(2, (these_indices.shape[0],), device=args.device)
                hal_idx = 0

                static = syn_static[static_idx, :, :, :]
                dynamic = syn_dynamic[label, dynamic_idx, :, :, :, :]
                hal = hals[hal_idx]

                x = hal(static, dynamic)
                this_y = label.long()

                if args.distributed:
                    forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    forward_params = student_params[-1]

                x = student_net(x, flat_param=forward_params)
                loss = criterion(x, this_y)

                grad = torch.autograd.grad(loss, student_params[-1], create_graph=True)[0]

                student_params.append(student_params[-1] - syn_lr * grad)

            param_loss = torch.tensor(0.0).to(args.device)
            param_dist = torch.tensor(0.0).to(args.device)

            param_loss += torch.nn.functional.mse_loss(student_params[-1], target_params, reduction="sum")
            param_dist += torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")

            param_loss_list.append(param_loss)
            param_dist_list.append(param_dist)

            param_loss /= num_params
            param_dist /= num_params

            param_loss /= param_dist

            grand_loss = param_loss

            if not args.no_train_static:
                optimizer_static.zero_grad()
            optimizer_dynamic.zero_grad()
            optimizer_hals.zero_grad()
            if args.train_lr:
                optimizer_lr.zero_grad()
            

            grand_loss.backward()

            if not args.no_train_static:
                optimizer_static.step()  
            optimizer_dynamic.step()
            optimizer_hals.step()
            if args.train_lr:
                optimizer_lr.step()
                syn_lr.data = syn_lr.data.clip(min=0.001)

            wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                "Grand_Loss/{}".format(start_epoch): grand_loss.detach().cpu(),
                "Start_Epoch": start_epoch})

            for _ in student_params:
                del _

            if it % 10 == 0:
                print('%s iter = %04d, param_loss = %.4f, param_dist = %.4f' % (get_time(), it, param_loss.item(), param_dist.item()))

    elif args.method == "DM":
        best_acc = {m: 0 for m in model_eval_pool}
        best_std = {m: 0 for m in model_eval_pool}
        
        for it in trange(0, args.Iteration + 1):
            if it%100 == 0:
                print("it:",it)
                print("syn_dynamic",dynamic_syn[0,0])
                for name, param in hals[-1].named_parameters():
                    print(name, param)

            save_this_it = False
            wandb.log({"Progress": it}, step=it)
            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                    args.model, model_eval, it))

                    accs_test = []
                    accs_train = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                            args.device)  # get a random model
                        
                        static_syn_eval = copy.deepcopy(static_syn.detach())
                        dynamic_syn_eval = copy.deepcopy(dynamic_syn.detach())
                        hal_eval = copy.deepcopy(hals)

                        args.lr_net = syn_lr.detach()
                        _, acc_train, acc_test, acc_per_cls = evaluate_synset(it_eval, net_eval,
                                                                [static_syn_eval, dynamic_syn_eval, hal_eval],
                                                                None, testloader, args, mode='multi-static')
                        accs_test.append(acc_test)
                        accs_train.append(acc_train)
                    accs_test = np.array(accs_test)
                    accs_train = np.array(accs_train)
                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    if acc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = acc_test_mean
                        best_std[model_eval] = acc_test_std
                        save_this_it = True
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                    len(accs_test), model_eval, acc_test_mean, acc_test_std))
                    wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                    wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                    wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                    wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

            if it in eval_it_pool and (save_this_it or it % 1000 == 0):
                with torch.no_grad():
                    image_save = static_syn.detach()
                    dynamic_save = dynamic_syn.flatten(0, 1).detach()

                    save_dir = os.path.join(args.save_path, project_name, wandb.run.name)

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    if not args.no_train_static:
                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                    torch.save(hals.state_dict(), os.path.join(save_dir, "hal_{}.pt".format(it)))
                    torch.save(dynamic_save.cpu(), os.path.join(save_dir, "dynamic_{}.pt".format(it)))

                    if save_this_it:
                        if not args.no_train_static:
                            torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt"))
                        torch.save(hals.state_dict(), os.path.join(save_dir, "weights_best.pt"))
                        torch.save(dynamic_save.cpu(), os.path.join(save_dir, "dynamic_best.pt"))
                        video_syn_eval = []
                        synset = MultiStaticSharedDataset(static_syn_eval, dynamic_syn_eval, hal_eval)
                        for i in range(50):
                            video_syn_eval.append(synset[i][0])
                        video_syn_eval = torch.stack(video_syn_eval, 0)
                        vis_shape = video_syn_eval.shape
                        video_syn_eval = video_syn_eval.view(vis_shape[0]*vis_shape[1],vis_shape[2],vis_shape[3],vis_shape[4])
                        for ch in range(3):
                            video_syn_eval[:,ch] = video_syn_eval[:,ch]*std[ch] + mean[ch]
                        video_syn_eval = torch.clamp(video_syn_eval,0,1)
                        save_name = os.path.join(save_dir, "syn_{}.png".format(it))
                        save_image(video_syn_eval, save_name, nrow=vis_shape[1])
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
            net.train()
            for param in list(net.parameters()):
                param.requires_grad = False

            embed = net.module.embed if args.distributed else net.embed

            loss_avg = 0
            
            label = torch.tensor(np.stack([np.ones(args.vpc)*i for i in range(0, num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
            ran = torch.arange(0, num_classes*args.vpc).to(args.device) # [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14, ..., ]
            idx = ran % args.vpc # [0,1,2,3,4,0,1,2,3,4, ..., 0,1,2,3,4]
            dynamic_idx = 2*idx + torch.randint(2, (num_classes * args.vpc,), device=args.device)
            static_idx = args.spc * label + 2*idx + torch.randint(2, (num_classes * args.vpc,), device=args.device)
            hal_idx = 0

            static = static_syn[static_idx, :, :, :]
            dynamic = dynamic_syn[label, dynamic_idx, :, :, :, :]
            hal = hals[hal_idx]
            image_syn = hal(static, dynamic)

            loss = torch.tensor(0.0).to(args.device)
            for c in range(0,num_classes):
                img_real = get_images(c, args.batch_real)
                img_syn = image_syn[c*args.vpc:(c+1)*args.vpc].reshape((args.vpc, args.frames, channel, im_size[0], im_size[1]))

                output_real = embed(img_real).detach()
                output_syn = embed(img_syn)

                loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

            if not args.no_train_static:
                optimizer_static.zero_grad()
            optimizer_dynamic.zero_grad()
            optimizer_hals.zero_grad()
            if args.train_lr:
                optimizer_lr.zero_grad()

            loss.backward()
            if not args.no_train_static:
                optimizer_static.step()  
            optimizer_dynamic.step()
            optimizer_hals.step()
            if args.train_lr:
                optimizer_lr.step()
                syn_lr.data = syn_lr.data.clip(min=0.001)

            loss_avg += loss.item()

            loss_avg /= (num_classes)


            wandb.log({"Loss": loss_avg}, step=it)
    else:
        raise NotImplementedError("Method {} not implemented".format(args.method))
    
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='dataset')

    parser.add_argument('--method', type=str, default='MTT', help='MTT or DC or DM')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')

    parser.add_argument('--spc', type=int, default=10, help='static memory(s) per class')
    parser.add_argument('--dpc', type=int, default=1, help='dynamic memory(s) per class')
    parser.add_argument('--vpc', type=int, default=5, help='')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=15000, help='how many distillation steps to perform')

    parser.add_argument('--no_train_static', action='store_true', help='do not train static memory')
    parser.add_argument('--path_static', type=str, default=None, help='path to pretrained static memory')
    parser.add_argument('--lr_static', type=float, default=100, help='learning rate for updating synthetic static memory')

    parser.add_argument('--lr_dynamic', type=float, default=0.01, help='learning rate for updating synthetic dynamic memory')

    parser.add_argument('--train_lr', action='store_true', help='train the learning rate')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_hal',type=float,default=0.01,help='learning rate for updating hallucinator') 

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--data_path', type=str, default='distill_utils/data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=64, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--preload', action='store_true',help="preload all data into RAM")
    
    parser.add_argument('--n_hal',type=int,default=1,help='number of hallucinators')
    parser.add_argument('--frames',type=int,default=16,help='number of frames')
    parser.add_argument('--num_workers',type=int,default=8,help='number of workers')
    parser.add_argument('--startIt',type=int,default=0,help='start iteration')
    parser.add_argument('--save_path',type=str, default='./logged_files', help='path to result')

    args = parser.parse_args()

    main(args)

