import os
import time

import logging

import matplotlib.pyplot as plt
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from lib_torch.networks import MLP, Conv, DC_ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN, \
    ResNet18_AP, ResNet18BN_AP, ConvNet3D, VideoConvNetMean, VideoConvNetMLP, VideoConvNetLSTM, VideoConvNetRNN, VideoConvNetGRU
from distill_utils.dataset import miniUCF101, staticUCF50, HMDB51


# @lru_cache()
def get_dataset(dataset, data_path, num_workers=0,img_size=(112,112),split_num=1,split_id=0,split_mode='mean'):
    if dataset == 'MNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FashionMNIST':
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.2861]
        std = [0.3530]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'SVHN':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
        dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'CIFAR100':
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes

    elif dataset == 'TinyImageNet':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        data = torch.load(os.path.join(data_path, 'tinyimagenet.pt'), map_location='cpu')

        class_names = data['classes']

        images_train = data['images_train']
        labels_train = data['labels_train']
        images_train = images_train.detach().float() / 255.0
        labels_train = labels_train.detach()
        for c in range(channel):
            images_train[:,c] = (images_train[:,c] - mean[c])/std[c]
        dst_train = TensorDataset(images_train, labels_train)  # no augmentation

        images_val = data['images_val']
        labels_val = data['labels_val']
        images_val = images_val.detach().float() / 255.0
        labels_val = labels_val.detach()

        for c in range(channel):
            images_val[:, c] = (images_val[:, c] - mean[c]) / std[c]

        dst_test = TensorDataset(images_val, labels_val)  # no augmentation
 
    elif dataset == 'ImageNet':
        channel = 3
        # im_size = (128, 128)
        im_size = (64, 64) 
        num_classes = 1000

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        resized_data_path = data_path+"/imagenet_%dx%d"%im_size
        if os.path.exists(resized_data_path):
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),])
            # the images are already resized
            path = resized_data_path
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            transforms.Resize(im_size),
                                            transforms.CenterCrop(im_size)])
            
            path = data_path+"/imagenet"

        dst_train = datasets.ImageNet(path, split="train", transform=transform) # no augmentation
        dst_test = datasets.ImageNet(path, split="val", transform=transform)
        class_names = None

    elif dataset == 'miniUCF101':
        # this is a video dataset, only 50 classes of UCF101
        channel = 3
        im_size = (112, 112)
        num_classes = 50

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/UCF101"
        print("miniUCF101 path:", path)
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            print("miniUCF im_size:", im_size)
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        dst_train = miniUCF101(path, split="train", transform=transform) # no augmentation
        dst_test  = miniUCF101(path, split="test", transform=transform)
        print("UCF101 train: ", len(dst_train), "test: ", len(dst_test))
        class_names = None
    
    elif dataset == 'HMDB51':
        # this is a video dataset
        channel = 3
        im_size = (112, 112)
        num_classes = 51

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/HMDB51"
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])

        dst_train = HMDB51(path, split="train", transform=transform) # no augmentation
        dst_test  = HMDB51(path, split="test", transform=transform)
        print("HMDB51 train: ", len(dst_train), "test: ", len(dst_test))
        class_names = None

    elif dataset == 'singleUCF50':
        # this is a video dataset, only get 1 frame of each video, 其中split_num和split_id调整选取范围, split_mode是选取帧的方式
        channel = 3
        im_size = (112, 112)
        num_classes = 50

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]  # use imagenet transform
        
        path = data_path+"/UCF101"
        assert os.path.exists(path)
        if im_size != (112,112):
            transform = transforms.Compose([transforms.Resize((100,80)),
                                            transforms.RandomCrop(im_size),
                                            #transforms.Resize(im_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)
                                            ])
        else:
            transform = transforms.Compose([#transforms.Resize((160,120)),
                                                #transforms.RandomCrop(im_size),
                                                #transforms.CenterCrop(im_size),
                                                #transforms.Resize(im_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=mean, std=std)
                                                ])

        dst_train = staticUCF50(path, split="train", transform=transform, frames = 1, split_num = 1, split_id = 0) # no augmentation
        dst_test  = staticUCF50(path, split="test", transform=transform, frames = 1, split_num = 1, split_id = 0)
        class_names = None
        
        print("UCF101 train: ", len(dst_train), "test: ", len(dst_test))

    else:
        exit('unknown dataset: %s' % dataset)

    testloader = torch.utils.data.DataLoader(dst_test, batch_size=64, shuffle=False, num_workers=num_workers)
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader


class TensorDataset(Dataset):
    def __init__(self, images, labels):  # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]

class HallucinatorSharedDataset(Dataset):
    def __init__(self, static, dynamic, hallucinator):
        self.static = static.detach().float()
        self.dynamic = dynamic.detach().float()
        self.hallucinator = hallucinator 
        self.n_c, _, _, _ = static.shape
        self.n_c, self.dpc, _, _, _, _ = dynamic.shape
    def __getitem__(self, index):
        n_pre_class = 1
        label = index // n_pre_class
        static = self.static[label, :, :, :] #3, 112, 112
        dynamic_idx = random.randint(0, self.dpc - 1)
        hal_idx = random.randint(0, len(self.hallucinator) - 1)
        dynamic = self.dynamic[label, dynamic_idx, :, :, :, :] #16, 1, 112, 112
        hallucinator = self.hallucinator[hal_idx]
        video = hallucinator(static.unsqueeze(0), dynamic.unsqueeze(0))
        return video[0], label #frames,c,h,w
    def __len__(self):
        return self.n_c 

class S1D1(Dataset):
    def __init__(self, static, dynamic, hallucinator, labels):
        self.static = static.detach().float()
        self.dynamic = dynamic.detach().float()
        self.hallucinator = hallucinator 
        self.n_s, _, _, _ = static.shape
        self.n_c, self.dpc, _, _, _, _ = dynamic.shape
        self.labels = labels.detach()
    def __getitem__(self, index):
        n_pre_class = self.n_s//self.n_c
        label = index // n_pre_class
        dynamic_idx = index % n_pre_class
        static = self.static[index, :, :, :] #3, 112, 112
        hal_idx = random.randint(0, len(self.hallucinator) - 1)
        dynamic = self.dynamic[label, dynamic_idx, :, :, :, :] #16, 1, 112, 112
        hallucinator = self.hallucinator[hal_idx]
        video = hallucinator(static.unsqueeze(0), dynamic.unsqueeze(0))
        return video[0], self.labels[index] #frames,c,h,w
    def __len__(self):
        return self.n_s

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instance', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling


def get_network(model, channel, num_classes, im_size=(32, 32), width=None, depth=None, norm=None):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if width is not None:
        net_width = width
    if depth is not None:
        net_depth = depth
    if norm is not None:
        net_norm = norm

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'DC_ConvNet':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'conv':
        net = Conv(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                   net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11(channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18AP':
        net = ResNet18_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)

    elif model == 'DC_ConvNetD1':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act,
                         net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetD2':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act,
                         net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetD3':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act,
                         net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetD4':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act,
                         net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetD5':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act,
                         net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvD3NN':
        net = Conv(channel=channel, num_classes=num_classes, net_width=128, net_depth=3, net_act='relu',
                   net_norm='none', net_pooling='avgpooling', im_size=im_size)
    elif model == 'ConvD3IN':
        net = Conv(channel=channel, num_classes=num_classes, net_width=128, net_depth=3, net_act='relu',
                   net_norm='instancenorm', net_pooling='avgpooling', im_size=im_size)
    elif model == 'ConvD3BN':
        net = Conv(channel=channel, num_classes=num_classes, net_width=128, net_depth=3, net_act='relu',
                   net_norm='batchnorm', net_pooling='avgpooling', im_size=im_size)
    elif model == 'ConvD3GN':
        net = Conv(channel=channel, num_classes=num_classes, net_width=128, net_depth=3, net_act='relu',
                   net_norm='groupnorm', net_pooling='avgpooling', im_size=im_size)

    elif model == 'ConvD4NN':
        net = Conv(channel=channel, num_classes=num_classes, net_width=64, net_depth=4, net_act='relu',
                   net_norm='none', net_pooling='avgpooling', im_size=(64, 64))
    elif model == 'ConvD4IN':
        net = Conv(channel=channel, num_classes=num_classes, net_width=64, net_depth=4, net_act='relu',
                   net_norm='instancenorm', net_pooling='avgpooling', im_size=(64, 64))

    elif model == 'DC_ConvNetD3NN':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=3, net_act='relu',
                         net_norm='none', net_pooling='avgpooling', im_size=im_size)
    elif model == 'DC_ConvNetD3IN':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=3, net_act='relu',
                         net_norm='instancenorm', net_pooling='avgpooling', im_size=im_size)
    elif model == 'DC_ConvNetD4NN':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=4, net_act='relu',
                         net_norm='none', net_pooling='avgpooling', im_size=(64, 64))
    elif model == 'DC_ConvNetD4IN':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=4, net_act='relu',
                         net_norm='instancenorm', net_pooling='avgpooling', im_size=(64, 64))

    elif model == 'DC_ConvNetW32':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act,
                         net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetW64':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act,
                         net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetW128':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act,
                         net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetW256':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act,
                         net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'DC_ConvNetAS':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetAR':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetAL':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'DC_ConvNetNN':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetBN':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetLN':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetIN':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetGN':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'DC_ConvNetNP':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=im_size)
    elif model == 'DC_ConvNetMP':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=im_size)
    elif model == 'DC_ConvNetAP':
        net = DC_ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth,
                         net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)
    elif model == 'ConvNet3D':
        net = ConvNet3D(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling='maxpooling', im_size=im_size,frames=16)

    else:
        net = None
        exit('unknown model: %s' % model)

    gpu_num = torch.cuda.device_count()
    if gpu_num > 0:
        device = 'cuda'
        if gpu_num > 1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net


def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))


def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4:  # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2:  # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1:  # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(
        1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis


def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec) ** 2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (
                torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s' % args.dis_metric)

    return dis


def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    elif ipc == 100:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc' % ipc)
    return outer_loop, inner_loop


def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].to(args.device)
        n_b = lab.shape[0]
        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(
            np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), np.argmax(lab.cpu().data.numpy(), axis=-1)))

        loss_avg += loss.item() * n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, mode='none', test_freq = None):
    net = net.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0005)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=int((Epoch + 1) * 0.1)),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Epoch + 1,
                                                   eta_min=args.lr_net * 0.01)])
    criterion = nn.MSELoss().to(args.device)

    if mode == 'none':
        images_train = images_train.to(args.device)
        labels_train = labels_train.to(args.device)
        dst_train = TensorDataset(images_train, labels_train)
    elif mode == 'hallucinator':
        dst_train = HallucinatorSharedDataset(images_train[0], images_train[1], images_train[2])
    elif mode == 'S1D1':
        labels_train = labels_train.to(args.device)
        dst_train = S1D1(images_train[0], images_train[1], images_train[2], labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)


    start = time.time()
    for ep in range(Epoch + 1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=False) # aug=False for video
        if (test_freq is None and ep == Epoch) or (test_freq is not None and ep % test_freq == 0):
            with torch.no_grad():
                loss_test, acc_test= epoch('test', testloader, net, optimizer, criterion, args, aug=False)
                print('Evaluate_%02d: Ep %d time = %ds loss = %.6f train acc = %.4f, test acc = %.4f' % (it_eval, ep, int(time.time() - start), loss_train, acc_train, acc_test))
        scheduler.step()
        # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        if ep % (Epoch // 10) == 0:
            print('Train epoch {}, acc = {}, loss = {}!'.format(ep, acc_train, loss_train))
    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (
        get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test


def mia_evaluate_synset(it_eval, net, images_train, labels_train, testloader, args, aug=True):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    # warmup_steps = 500
    # warmup_fn = optax.linear_schedule(init_value=0., end_value=lr, transition_steps=warmup_steps)
    # cosine_fn = optax.cosine_decay_schedule(init_value=lr,
    #                                         decay_steps=max(Epoch - warmup_steps, 1))
    # learning_rate_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn], boundaries=[warmup_steps])
    # print('step per epoch {}'.format(len(trainloader)))

    start = time.time()
    for ep in range(Epoch + 1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, args, aug=aug)
        # if ep in lr_schedule:
        #     lr *= 0.1
        #     # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        #     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        # lr = learning_rate_fn(len(trainloader) * (ep + 1))
        # optimizer = torch.optim.Adam(net.parameters(), lr=float(lr))

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
    print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (
        get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test


def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:, c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1], shape[2] + crop * 2, shape[3] + crop * 2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop + shape[2], crop:crop + shape[3]] = images[i]
            r, c = np.random.permutation(crop * 2)[0], np.random.permutation(crop * 2)[0]
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1),
                              cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)

        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0]  # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images


def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['DC_ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M':  # multiple architectures
        model_eval_pool = ['MLP', 'DC_ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'CA':  # multiple architectures
        model_eval_pool = ['MLP', 'DC_ConvNet', 'ConvD3NN', 'ConvD3IN',
                           'DC_ConvNetD3NN', 'DC_ConvNetD3IN', 'LeNet', 'AlexNet', 'VGG11', 'VGG11BN', 'ResNet18',
                           'ResNet18AP', 'ResNet18BN']
    elif eval_mode == 'W':  # ablation study on network width
        model_eval_pool = ['DC_ConvNetW32', 'DC_ConvNetW64', 'DC_ConvNetW128', 'DC_ConvNetW256']
    elif eval_mode == 'D':  # ablation study on network depth
        model_eval_pool = ['DC_ConvNetD1', 'DC_ConvNetD2', 'DC_ConvNetD3', 'DC_ConvNetD4']
    elif eval_mode == 'A':  # ablation study on network activation function
        model_eval_pool = ['DC_ConvNetAS', 'DC_ConvNetAR', 'DC_ConvNetAL']
    elif eval_mode == 'P':  # ablation study on network pooling layer
        model_eval_pool = ['DC_ConvNetNP', 'DC_ConvNetMP', 'DC_ConvNetAP']
    elif eval_mode == 'N':  # ablation study on network normalization layer
        model_eval_pool = ['DC_ConvNetNN', 'DC_ConvNetBN', 'DC_ConvNetLN', 'DC_ConvNetIN', 'DC_ConvNetGN']
    elif eval_mode == 'S':  # itself
        if 'BN' in model:
            print(
                'Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    elif eval_mode == 'DNFRD3':
        model_eval_pool = ['ConvD3NN', 'ConvD3IN', 'DC_ConvNetD3NN', 'DC_ConvNetD3IN']
    elif eval_mode == 'DNFRD4':
        model_eval_pool = ['ConvD4NN', 'ConvD4IN', 'DC_ConvNetD4NN', 'DC_ConvNetD4IN']
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S'  # 'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5  # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5
        self.Siamese = False


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed=-1, param=None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M':  # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s' % param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0 / ratio) + 1.0 / ratio
    theta = [[[sx[i], 0, 0],
              [0, sy[i], 0], ] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese:  # Siamese augmentation:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param):  # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
              [torch.sin(theta[i]), torch.cos(theta[i]), 0], ] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese:  # Siamese augmentation:
        theta[:] = theta[0].clone()
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randf[:] = randf[0].clone()
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0].clone()
    x = x + (randb - 0.5) * ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0].clone()
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0].clone()
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0].clone()
        translation_y[:] = translation_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0].clone()
        offset_y[:] = offset_y[0].clone()
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}


def save_torch_image(x_proto, y_proto, step, num_classes=10, class_names=None, rev_preprocess_op=None, image_dir=None,
                     is_grey=False, save_np=False, save_img=False, scale=None):
    def scale_for_vis(img, rev_preprocess_op=None):
        if rev_preprocess_op:
            img = rev_preprocess_op(img)
        else:
            img = img / img.std() * 0.2 + 0.5
        img = np.clip(img, 0, 1)
        return img

    x_proto = np.transpose(x_proto, axes=[0, 2, 3, 1])

    if save_np and image_dir:
        path = os.path.join(image_dir, 'np')
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
        logging.info('Save prototype to numpy! Path: {}'.format(save_path))
        np.savez('{}.npz'.format(save_path), image=x_proto, label=y_proto)

    x_proto = scale_for_vis(x_proto, rev_preprocess_op)
    y_proto = np.argmax(y_proto, axis=-1)

    total_images = y_proto.shape[0]
    total_index = list(range(total_images))
    total_img_per_class = total_images // num_classes
    img_per_class = 100 // num_classes

    if num_classes <= 100:
        select_idx = []
        # always select the top to make it consistent
        for i in range(num_classes):
            select = total_index[i * total_img_per_class: (i + 1) * total_img_per_class][:img_per_class]
            select_idx.extend(select)
    else:
        select_idx = []
        # always select the top to make it consistent
        for i in range(100):
            select = total_index[i * total_img_per_class: (i + 1) * total_img_per_class][0]
            select_idx.append(select)

    row, col = len(select_idx) // 10, 10
    fig = plt.figure(figsize=(33, 33))
    fig.patch.set_facecolor('black')
    for i, idx in enumerate(select_idx[: row * col]):
        img = x_proto[idx]
        ax = plt.subplot(row, col, i + 1)
        if class_names is not None:
            ax.set_title('{} ({})'.format(class_names[y_proto[idx]], y_proto[idx]), x=0.5, y=0.9,
                         backgroundcolor='silver')
        else:
            ax.set_title('class_{}'.format(y_proto[idx]), x=0.5, y=0.9, backgroundcolor='silver')

        if is_grey:
            plt.imshow(np.squeeze(img), cmap='gray')
        else:
            plt.imshow(np.squeeze(img))

        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        plt.imshow(np.squeeze(img))
        plt.xticks([])
        plt.yticks([])

    fig.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    if save_img and image_dir:
        path = os.path.join(image_dir, 'png')
        if not os.path.exists(path):
            os.mkdir(path)
        if scale is not None:
            save_path = os.path.join(path, 'step{}_ss{}'.format(str(step).zfill(6), scale))
        else:
            save_path = os.path.join(path, 'step{}'.format(str(step).zfill(6)))
        logging.info('Save prototype to numpy! Path: {}'.format(save_path))
        fig.savefig('{}.png'.format(save_path), bbox_inches='tight')

    return fig

class Conv3DNet(nn.Module):
    def __init__(self, in_channel=4, mid_channel=3, out_channel=3, img_size=112, kernel_size=3, mode='concat'):
        super().__init__()
        self.mode = mode
        if mode == 'add':
            in_channel = 3
        self.encoder = nn.Conv3d(in_channel, mid_channel, kernel_size, padding=1)

    def forward(self, static, dynamic):
        b, f, _, h, w = dynamic.shape # bz, 16, 1, 112, 112
        static = static.repeat(f, 1, 1, 1, 1).permute(1, 2, 0, 3, 4) #bz, 3, 16, h, w
        dynamic = dynamic.permute(0, 2, 1, 3, 4) #bz, 1, 16, h, w
        if self.mode == 'concat':
            x = torch.cat([static, dynamic], dim=1) #bz, 4, f, h, w
        elif self.mode == 'add':
            x = static + dynamic #bz, 3, f, h, w
        else:
            raise NotImplementedError
        x = self.encoder(x)
        return x.permute(0, 2, 1, 3, 4) 