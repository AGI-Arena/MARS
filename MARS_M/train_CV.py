"""
Adapted from uclaml/Padam: https://github.com/uclaml/Padam/blob/master/run_cnn_test_cifar10.py
"""
import numpy as np
import os
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["mnist", "cifar10", "cifar100"], help="dataset to use"
)
parser.add_argument(
    "--scheduler", type=str, default="multistep", choices=["multistep", "cosine", "constant"], help="scheduler to use"
)
parser.add_argument("--train_bsz", type=int, default=128, help="training batch size")
parser.add_argument("--eval_bsz", type=int, default=100, help="eval batch size")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")
parser.add_argument("--cuda", type=str, default="0", help="device to use")

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--adamw_lr', default=0.003, type=float, help='learning rate for adamw')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--optim', '-m', type=str, choices=["adam", "adamw", "mars-m", "moonlight"], default='mars', help='optimization method, default: mars')
parser.add_argument('--net', '-n', type=str, default="resnet18", help='network archtecture, choosing from "simple_cnn" or torchvision models. default: resnet18')
parser.add_argument('--wd', default=0., type=float, help='weight decay')
parser.add_argument('--Nepoch', default=200, type=int, help='number of epoch')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
parser.add_argument('--gamma', default=0.025, type=float, help='gamma in MARS-M')
parser.add_argument('--clip_c', action='store_true', help='whether to clip c_t in MARS-M')
parser.add_argument('--mars_exact', action='store_true', help='whether to use the approximate version of MARS-M')
parser.add_argument('--wandb', action='store_true', help='use wandb')
parser.add_argument('--save_dir', type=str, default="./checkpoint", help='save directory')
parser.add_argument('--wandb_name', type=str, default="None", help='log directory')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
if args.wandb:
    import wandb
    if args.wandb_name == "None":
        wandb.init(project="CV", name=args.dataset+"_"+args.net+"_"+args.optim+"_"+str(args.lr), config=args)
    else:
        wandb.init(project="CV", name=args.wandb_name, config=args)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.cv_utils import get_datasets, get_scheduler, get_model
use_cuda = torch.cuda.is_available() and not args.cpu

os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

trainloader, testloader = get_datasets(args.dataset, args.train_bsz, args.eval_bsz)
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.net}_{args.dataset}_'+args.optim)
    model = checkpoint['model']
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    train_errs = checkpoint['train_errs']
    test_errs = checkpoint['test_errs']
else:
    print('==> Building model..')

model = get_model(args)

if use_cuda:
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
 
    
criterion = nn.CrossEntropyLoss()

betas = (args.beta1, args.beta2)
from optimizers.mars_m import MARS_M
from optimizers.moonlight import Moonlight
if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd, betas = betas)
elif args.optim == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.wd, betas = betas)
elif args.optim == 'muon':
    muon_params = [p for p in model.parameters() if p.ndim == 2]
    adamw_params = [p for p in model.parameters() if p.ndim != 2]
    optimizer = Moonlight(muon_params=muon_params, adamw_params=adamw_params, 
                          lr=args.lr, wd=args.wd, adamw_betas=betas)
elif args.optim == 'mars-m':
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    adamw_params = [p for p in model.parameters() if p.ndim < 2]
    if args.mars_exact:
        is_approx = False
    else:
        is_approx = True
    optimizer = MARS_M(muon_params=muon_params, adamw_params=adamw_params, 
                          lr=args.lr, wd=args.wd, adamw_betas=betas, gamma=args.gamma, 
                          clip_c=args.clip_c, is_approx=is_approx)
    
scheduler = get_scheduler(optimizer, args)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_errs = []
test_errs = []
train_losses = []
test_losses = []
acc_list = []
t_bar = tqdm(total=len(trainloader))
t_bar2 = tqdm(total=len(testloader))
previous_input = None
previous_target = None
for epoch in range(start_epoch+1, args.Nepoch+1):
    
    scheduler.step()
    # print ('\nEpoch: %d' % epoch, ' Learning rate:', scheduler.get_lr())
    model.train()  # Training

    train_loss = 0
    correct_train = 0
    total_train = 0
    t_bar.reset()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.mars_exact:
            if previous_input is not None:
                if use_cuda:
                    p_inputs, p_targets = previous_input.cuda(), previous_target.cuda()
                optimizer.zero_grad()
                p_outputs = model(p_inputs)
                p_loss = criterion(p_outputs, p_targets)
                p_loss.backward()
                optimizer.update_previous_grad()
            else:
                # copy inputs to previous_input
                previous_input = inputs.clone()
                previous_target = targets.clone()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if args.mars_exact:
            optimizer.update_last_grad()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += predicted.eq(targets.data).cpu().sum().item()

        t_bar.update(1)
        t_bar.set_description('Epoch: %d | Loss: %.3f | Acc: %.3f%% ' % (epoch, train_loss/(batch_idx+1), 100.0/total_train*(correct_train)))
        t_bar.refresh()
    train_losses.append(train_loss/(batch_idx+1))
    train_errs.append(1 - correct_train/total_train)

    model.eval() # Testing
 
    test_loss = 0
    correct = 0
    total = 0
    t_bar2.reset()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        t_bar2.update(1)
        t_bar2.set_description('Loss: %.3f | Acc: %.3f%% (Best: %.3f%%)' % (test_loss/(batch_idx+1), 100.0/total*(correct), best_acc))
        t_bar2.refresh()
    test_errs.append(1 - correct/total)
    test_losses.append(test_loss/(batch_idx+1))
    if args.wandb:
        wandb.log({"epoch": epoch,
               "train_loss": train_loss/(batch_idx+1), 
               "train_acc": 100.0/total_train*(correct_train), 
               "test_loss": test_loss/(batch_idx+1), 
               "test_acc": 100.0/total*(correct),
               "lr": scheduler.get_lr()[0]}, step=epoch)
    # Save checkpoint
    acc = 100.0/total*(correct)
    if acc > best_acc:
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state = {
                'model': model,
                'epoch': epoch,
                }
        # torch.save(state, './checkpoint/cnn_cifar10_' + args.optim)
        torch.save(state, os.path.join(args.save_dir, "-".join([args.net, args.dataset, args.optim, str(args.lr).replace(".", "_")])+".pth"))
        best_acc = acc
        t_bar2.set_description('Model Saved! | Loss: %.3f | Acc: %.3f%% (Best: %.3f%%)' % (test_loss/(batch_idx+1), 100.0/total*(correct), best_acc))
        t_bar2.refresh()
    acc_list.append(acc) 
     