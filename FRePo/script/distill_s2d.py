import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch._vmap_internals import vmap
from tqdm import tqdm, trange
import numpy as np
import os
import time
import  random
import wandb

os.environ['JAX_PLATFORM_NAME'] = 'cpu'
sys.path.append("..")

from lib_torch.utils import get_network, evaluate_synset, get_dataset, get_eval_pool, HallucinatorSharedDataset, Conv3DNet

@vmap
def lb_margin_th(logits):
    dim = logits.shape[-1]
    val, idx = torch.topk(logits, k=2)
    margin = torch.minimum(val[0] - val[1], torch.tensor(1 / dim, dtype=torch.float, device=logits.device))
    return -margin


class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

class SynData(nn.Module):
    def __init__(self, x_init, y_init, learn_label=False):
        super(SynData, self).__init__()
        self.x_syn = nn.Parameter(x_init, requires_grad=True)
        self.y_syn = nn.Parameter(y_init, requires_grad=learn_label)

    def forward(self):
        return self.x_syn, self.y_syn

    def value(self):
        '''Return the synthetic images and labels. Used in deterministic parameterization of synthetic data'''
        return self.x_syn.detach(), self.y_syn.detach()

class S2DSynData(nn.Module):
    def __init__(self, static, dynamic, hallucinator, y_init,learn_label=False):
        super(S2DSynData, self).__init__()
        self.static = nn.Parameter(static, requires_grad=False)
        self.dynamic = nn.Parameter(dynamic, requires_grad=True)
        self.hallucinator = hallucinator
        self.y_syn = nn.Parameter(y_init, requires_grad=learn_label)
        self.n_c, self.dpc, _, _, _, _ = dynamic.shape
        self.x_syn = None
    def forward(self):
        x_syn = []
        for i in range(self.n_c * self.dpc):
            hal_idx = random.randint(0, len(self.hallucinator) - 1)
            dynamic_idx = i % self.dpc
            static_idx = i 
            hal = self.hallucinator[hal_idx]
            dyn = self.dynamic[i//self.dpc, dynamic_idx, :, :, :, :]
            sta = self.static[static_idx, :, :, :]
            x = hal(sta.unsqueeze(0), dyn.unsqueeze(0))
            x_syn.append(x[0])
        self.x_syn = torch.stack(x_syn)
        return self.x_syn, self.y_syn
    
    def value(self):
        '''Return the synthetic images and labels. Used in deterministic parameterization of synthetic data'''
        self.forward()
        return self.x_syn.detach(), self.y_syn.detach()

            

class PoolElement():
    def __init__(self, get_model, get_optimizer, get_scheduler, loss_fn, batch_size, max_online_updates, idx, device,
                 step=0):
        self.get_model = get_model
        self.get_optimizer = get_optimizer
        self.get_scheduler = get_scheduler
        self.loss_fn = loss_fn.to(device)
        self.batch_size = batch_size
        self.max_online_updates = max_online_updates
        self.idx = idx
        self.device = device
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.initialize()
        self.step = step

    def __call__(self, x, no_grad=False):
        self.model.eval()
        if no_grad:
            with torch.no_grad():
                return self.model(x)
        else:
            return self.model(x)

    def feature(self, x, no_grad=False, weight_grad=False):
        self.model.eval()
        if no_grad:
            with torch.no_grad():
                return self.model.embed(x)
        else:
            self.model.requires_grad_(weight_grad)
            return self.model.embed(x)

    def nfr(self, x_syn, y_syn, x_tar, reg=1e-6, weight_grad=False, use_flip=False):
        if use_flip:
            x_syn_flip = torch.flip(x_syn, dims=[-1])
            x_syn = torch.cat((x_syn, x_syn_flip), dim=0)
            y_syn = torch.cat((y_syn, y_syn), dim=0)

        feat_tar = self.feature(x_tar, no_grad=True)
        feat_syn = self.feature(x_syn, weight_grad=weight_grad)

        kss = torch.mm(feat_syn, feat_syn.t())
        kts = torch.mm(feat_tar, feat_syn.t())
        kss_reg = (kss + np.abs(reg) * torch.trace(kss) * torch.eye(kss.shape[0], device=kss.device) / kss.shape[0])
        pred = torch.mm(kts, torch.linalg.solve(kss_reg, y_syn))
        return pred

    def nfr_eval(self, feat_syn, y_syn, x_tar, kss_reg):
        feat_tar = self.feature(x_tar, no_grad=True)
        kts = torch.mm(feat_tar, feat_syn.t())
        pred = torch.mm(kts, torch.linalg.solve(kss_reg, y_syn))
        return pred

    def train_steps(self, x_syn, y_syn, steps=1):
        self.model.train()
        self.model.requires_grad_(True)
        for step in range(steps):
            x, y = self.get_batch(x_syn, y_syn)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(x)
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        self.check_for_reset(steps=steps)

    def evaluate_syn(self, x_syn, y_syn):
        pass

    def get_batch(self, xs, ys):
        if ys.shape[0] < self.batch_size:
            x, y = xs, ys
        else:
            sample_idx = np.random.choice(ys.shape[0], size=(self.batch_size,), replace=False)
            x, y = xs[sample_idx], ys[sample_idx]
        return x, y

    def initialize(self):
        self.model = self.get_model().to(self.device)
        self.optimizer = self.get_optimizer(self.model)
        self.scheduler = self.get_scheduler(self.optimizer)
        self.step = 0

    def check_for_reset(self, steps=1):
        self.step += steps
        if self.step >= self.max_online_updates:
            self.initialize()


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader= get_dataset(args.dataset, args.data_path)
    y_train =  dst_train.labels
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = dst_test.labels
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    eval_it_pool = np.arange(args.startIt, args.Iteration + 1, args.eval_it).tolist()
    print('Evaluation iterations: ', eval_it_pool)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    num_prototypes = args.num_prototypes_per_class * num_classes

    steps_per_eval = 10000
    steps_per_save = 400

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
    
    project_name = "S2D_{}".format(args.method)

    wandb.init(sync_tensorboard=False,
               project=project_name,
               job_type="CleanRepo",
               config=args,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    #step_per_prototpyes = {10: 1000, 100: 2000, 200: 20000, 400: 5000, 500: 5000, 1000: 10000, 2000: 40000, 5000: 40000}
    #num_online_eval_updates = step_per_prototpyes[num_prototypes]

    args.batch_train = min(num_prototypes, 500) if args.batch_train is None else args.batch_train
    criterion = nn.MSELoss(reduction='none').to(args.device)

    ''' organize the real dataset '''
    labels_all = dst_train.labels
    indices_class = [[] for c in range(num_classes)]

    print("BUILDING DATASET")
    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    def get_images(c, n):  # get random n images from class c
        # import pdb; pdb.set_trace()
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        if n == 1:
            imgs = dst_train[idx_shuffle[0]][0].unsqueeze(0)
        else:
            imgs = torch.cat([dst_train[i][0].unsqueeze(0) for i in idx_shuffle], 0)
        return imgs.to(args.device)

    static_syn = torch.randn(size=(num_classes*args.num_prototypes_per_class, 3, im_size[0], im_size[1]), dtype=torch.float)
    dynamic_syn = torch.randn(size=(num_classes, args.dpc, args.frames, 1, im_size[0], im_size[1]), dtype=torch.float)

    ''' initialize the hallucinator '''
    hals = nn.ModuleList([Conv3DNet() for _ in range(args.n_hal)])
    y_syn = torch.tensor(np.stack([np.ones(args.num_prototypes_per_class)*i for i in range(0, num_classes)]), dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.path_static is not None:
        static_syn = torch.load(args.path_static)["image"]
        print('load static memory from %s' % args.path_static)
        print('static_syn shape: ', static_syn.shape)

    y_scale = np.sqrt(num_classes / 10)
    y_train = F.one_hot(y_train, num_classes=num_classes) - 1 / num_classes
    y_test = F.one_hot(y_test, num_classes=num_classes) - 1 / num_classes

    dst_train.labels = y_train 
    dst_test.labels = y_test

    trainloader = InfiniteDataLoader(dst_train, batch_size=512, shuffle=True, num_workers=args.num_workers) #1024->512
    testloader = DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=args.num_workers)

    y_syn = (F.one_hot(y_syn, num_classes=num_classes) - 1 / num_classes)/ y_scale


    syndata = S2DSynData(static_syn,dynamic_syn, hals, y_syn, learn_label=args.learn_label).to(args.device)

    d_params = [param for name, param in syndata.named_parameters() if 'dynamic' in name]
    other_params = [param for name, param in syndata.named_parameters() if 'dynamic' not in name]
    synopt = torch.optim.Adam([{"params":d_params,"lr":args.lr_d},{"params":other_params,"lr":args.lr_h}])
    #synopt = torch.optim.Adam(syndata.parameters(), lr=args.lr_h)
    synsch = torch.optim.lr_scheduler.CosineAnnealingLR(synopt, T_max=args.Iteration, eta_min=args.lr_h * 0.1)

    step_offset = 0
    best_val_acc = 0.0

    loss_sum = 0.0
    ln_loss_sum = 0.0
    lb_loss_sum = 0.0
    count = 0
    last_t = time.time()

    get_model = lambda: get_network(args.model, channel, num_classes, im_size).to(args.device)
    get_optimizer = lambda m: torch.optim.Adam(m.parameters(), lr=args.lr_net, betas=(0.9, 0.999))
    get_scheduler = lambda o: torch.optim.lr_scheduler.ChainedScheduler([
        torch.optim.lr_scheduler.LinearLR(o, start_factor=0.01, total_iters=500),
        torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=args.max_online_updates, eta_min=args.lr_net * 0.01)])

    pools = []
    for idx in range(args.num_nn_state):
        init_step = (args.max_online_updates // args.num_nn_state) * idx
        pools.append(PoolElement(get_model=get_model, get_optimizer=get_optimizer, get_scheduler=get_scheduler,
                                 loss_fn=nn.MSELoss(), batch_size=500, max_online_updates=args.max_online_updates, idx=idx,
                                 device=args.device, step=init_step))
        
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    for it in range(step_offset + 1, args.Iteration + 1):
        ''' Evaluate synthetic data '''
        # if (it-1) % 100 == 0:
        #     print("it:",it)
        #     print("syn_dynamic",syndata.dynamic[0,0])
        #     for name, param in syndata.hallucinator[0].named_parameters():
        #         print(name, param)

        if it in eval_it_pool or it % steps_per_eval == 0:
            for model_eval in model_eval_pool:
                print('Evaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                accs_test = []
                accs_train = []
                for it_eval in range(args.num_eval):
                    net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                    image_syn_eval, label_syn_eval = syndata.value() # avoid any unaware modification
                    _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args, test_freq=250)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                    len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)

        if it % steps_per_save == 0 or it == 1:
            ''' save synthetic data '''
            dynamic_save = syndata.dynamic.detach().cpu()
            hals_save = syndata.hallucinator
            y_save = syndata.y_syn.detach().cpu()
            save_dir = os.path.join(".", "logged_files", project_name, wandb.run.name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(dynamic_save, os.path.join(save_dir, "dynamic_%d.pt"%it))
            torch.save(hals_save.state_dict(), os.path.join(save_dir, "hal_%d.pt"%it))
            torch.save(y_save, os.path.join(save_dir, "y_syn_%d.pt"%it))
        x_target, y_target = next(trainloader)
        x_target = x_target.to(args.device)
        y_target = y_target.to(args.device)
        x_syn, y_syn = syndata()

        idx = np.random.randint(low=0, high=args.num_nn_state)
        pool_m = pools[idx]

        if torch.cuda.device_count() > 1:
            pool_m.model = pool_m.model.module

        y_pred = pool_m.nfr(x_syn, y_syn, x_target, use_flip=False) # dont use flip
        ln_loss = criterion(y_pred, y_target).sum(dim=-1).mean(0)
        lb_loss = lb_margin_th(y_syn).mean()
        loss = ln_loss + lb_loss

        synopt.zero_grad(set_to_none=True)
        loss.backward()
        synopt.step()
        x_syn, y_syn = syndata.value()
        pool_m.train_steps(x_syn, y_syn, steps=1)
        
        wandb.log({"Loss": loss.detach().cpu()}, step=it)

        synsch.step()
        loss_sum += loss.item() * x_target.shape[0]
        ln_loss_sum += ln_loss.item() * x_target.shape[0]
        lb_loss_sum += lb_loss.item() * x_target.shape[0]
        count += x_target.shape[0]

        if it % 100 == 0:
            '''x_syn, y_syn = syndata.value()
            x_norm = torch.mean(torch.linalg.norm(x_syn.view(x_syn.shape[0], -1), ord=2, dim=-1)).cpu().numpy()
            y_norm = torch.mean(torch.linalg.norm(y_syn.view(y_syn.shape[0], -1), ord=2, dim=-1)).cpu().numpy()
            summary = {'train/loss': loss_sum / count,
                       'train/ln_loss': ln_loss_sum / count,
                       'train/lb_loss': lb_loss_sum / count,
                       'monitor/steps_per_second': count / 1024 / (time.time() - last_t),
                       'monitor/learning_rate': synsch.get_last_lr()[0],
                       'monitor/x_norm': x_norm,
                       'monitor/y_norm': y_norm}
            writer.write_scalars(it, summary)'''
            last_t = time.time()
            loss_sum, ln_loss_sum, lb_loss_sum, count = 0.0, 0.0, 0.0, 0

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='miniUCF101', help='dataset name')
    parser.add_argument('--method', type=str, default='FRePo', help='MTT or DC or FRePo')
    parser.add_argument('--model', type=str, default='ConvNet3D', help='model')

    parser.add_argument('--num_prototypes_per_class', type=int, default=1, help='image(s) per class')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=1000, help='how often to evaluate')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')


    parser.add_argument('--epoch_eval_train', type=int, default=500,
                        help='epochs to train a model with synthetic data')
    parser.add_argument('--batch_train', type=int, default=None, help='batch size for training')
    parser.add_argument('--batch_syn', type=int, default=512, help='batch size for syn')
    parser.add_argument('--init', type=str, default='real', choices=['noise', 'real', 'real-all'], help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--data_path', type=str, default='/hdd/DATA/video_distill/distill_utils/data', help='dataset path')

    parser.add_argument('--startIt', type=int, default=0, help='test start iteration')
    parser.add_argument('--Iteration', type=int, default=10000, help='test end iteration')
    parser.add_argument('--max_online_updates', type=int, default=100, help='max online updates')
    parser.add_argument('--num_nn_state', type=int, default=10, help='number of neural network states')
    parser.add_argument('--learn_label', action='store_true', help='learn label or not')
    parser.add_argument('--lr_d', type=float, default=1e2, help='learning rate for synthetic data')
    parser.add_argument('--lr_h', type=float, default=0.001, help='learning rate for hallucinator')
    parser.add_argument('--lr_net', type=float, default=0.001, help='learning rate for neural network')
    parser.add_argument('--num_workers', type=int, default=16, help='number of workers for data loading')
    parser.add_argument('--path_static', type=str, default=None, help='path to static memory')
    parser.add_argument('--dpc', type=int, default=1, help='dpc')
    parser.add_argument('--spc', type=int, default=1, help='spc')
    parser.add_argument('--n_hal', type=int, default=1, help='number of hallucinators')
    parser.add_argument('--frames', type=int, default=16, help='number of frames')


    args = parser.parse_args()

    main(args)

