import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm, trange
from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug
import wandb

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

        video_all = []
        label_all = []
        for i in trange(len(dst_test)):
            _ = dst_test[i]
            video_all.append(_[0])
            label_all.append(_[1])
        video_all = torch.stack(video_all)
        label_all = torch.tensor(label_all)
        dst_test = torch.utils.data.TensorDataset(video_all, label_all)
        testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_train, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    project_name = "Buffer"

    wandb.init(sync_tensorboard=False,
               project=project_name,
               job_type="CleanRepo",
               config=args,
               )
    
    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    save_dir = args.buffer_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    criterion = nn.CrossEntropyLoss().to(args.device)

    trajectories = []

    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=args.num_workers)

    for it in range(0, args.num_experts):
        ''' Train synthetic data '''
        teacher_net = get_network(args.model, channel, num_classes,im_size).to(args.device) # get a random model
        teacher_net.train()
        lr = args.lr_teacher
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data
        teacher_optim.zero_grad()

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]

        for e in trange(args.train_epochs, postfix=str(it), ncols=60):

            train_loss, train_acc, _ = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        criterion=criterion, args=args)

            if e % 10 == 0 or e == args.train_epochs - 1:
                test_loss, test_acc,acc_per_cls= epoch("test", dataloader=testloader, net=teacher_net, optimizer=None, 
                                            criterion=criterion, args=args)
                print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it, e, train_acc, test_acc))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()
        print("acc_per_cls: ", acc_per_cls)
        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.001, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--num_workers', type=int, default=8, help='')


    parser.add_argument('--data_path', type=str, default='distill_utils/data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./logs/buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)

    parser.add_argument('--preload', action='store_true', help='preload dataset to memory')

    args = parser.parse_args()
    main(args)


