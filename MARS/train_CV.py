"""
Adapted from uclaml/Padam: https://github.com/uclaml/Padam/blob/master/run_cnn_test_cifar10.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import os
import argparse
import json

from utils.cv_utils import get_datasets, get_scheduler, get_model
from tqdm import tqdm

        
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"], help="dataset to use"
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
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--optim', '-m', type=str, choices=["adam", "adamw", "mars", "muon"], default='mars', help='optimization method, default: mars')
parser.add_argument('--net', '-n', type=str, default="resnet18", help='network archtecture, choosing from "simple_cnn" or torchvision models. default: resnet18')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--Nepoch', default=200, type=int, help='number of epoch')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
parser.add_argument('--wandb', action='store_true', help='use wandb')
parser.add_argument('--save_dir', type=str, default="./checkpoint", help='save directory')


args = parser.parse_args()
if args.wandb:
    import wandb
    wandb.init(project="CV", name=args.dataset+"_"+args.optim+"_"+str(args.lr), config=args)
use_cuda = torch.cuda.is_available() and not args.cpu
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
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
    checkpoint = torch.load('./checkpoint/cnn_cifar10_'+args.optim)
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
from optimizers.mars import MARS
from optimizers.muon import Muon
from opt import CombinedOptimizer
from optimizers.adamw import AdamW
if args.optim == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.wd, betas = betas)
elif args.optim == 'adamw':
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.wd, betas = betas)
elif args.optim == 'muon':
    optimizer = CombinedOptimizer(model.parameters(), [AdamW, Muon], [{'lr': 0.1, 'betas': betas, 'weight_decay': args.wd},
                                                      {'lr': args.lr, 'weight_decay': 0.}])
elif args.optim == 'mars':
    optimizer = MARS(model.parameters(), lr=args.lr, weight_decay = args.wd)
    
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
for epoch in range(start_epoch+1, args.Nepoch+1):
    
    scheduler.step()
    # print ('\nEpoch: %d' % epoch, ' Learning rate:', scheduler.get_lr())
    model.train()  # Training

    train_loss = 0
    correct = 0
    total = 0
    
    t_bar.reset()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        t_bar.update(1)
        t_bar.set_description('Epoch: %d | Loss: %.3f | Acc: %.3f%% ' % (epoch, train_loss/(batch_idx+1), 100.0/total*(correct)))
        t_bar.refresh()
    train_losses.append(train_loss/(batch_idx+1))
    train_errs.append(1 - correct/total)

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
               "train_acc": 100.0/total*(correct), 
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
     