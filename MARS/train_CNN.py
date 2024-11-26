# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import List, Tuple, Type

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from utils.model_CNN import Network
from optimizers.adopt import ADOPT
from optimizers.mars import MARS
import random
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument(
    "--dataset", type=str, default="cifar10", choices=["mnist", "cifar10"], help="dataset to use"
)
parser.add_argument("-b", "--batch_size", type=int, default=128, help="batch size")
parser.add_argument("-e", "--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--cpu", action="store_true", help="use cpu only")


def get_datasets(dataset_name: str, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Get train and test dataloaders."""
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    elif dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    else:
        raise NotImplementedError(f"{dataset_name=} is not implemented.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader


class WarmupCosineScheduler:
    """Custom learning rate scheduler with linear warmup and cosine decay."""
    def __init__(self, optimizer, warmup_iters: int, total_iters: int, min_lr: float, max_lr: float):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_iter = 0
        self.lr = 0
        
    def step(self):
        self.current_iter += 1
        if self.current_iter <= self.warmup_iters:
            lr = self.current_iter / self.warmup_iters * self.max_lr
        else:
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                np.cos((self.current_iter - self.warmup_iters) / (self.total_iters - self.warmup_iters) * 3.14159265 / 2)
            ).item()
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.lr = lr

class Trainer:
    """Training manager for PyTorch models."""
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, device: torch.device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc_trace = []
        self.val_acc_trace = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        correct = 0
        total = 0
        
        for batch in train_loader:
            images, targets = batch[0].to(self.device), batch[1].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if self.scheduler is not None:
                self.scheduler.step()
        return 100. * correct / total
    
    def evaluate(self, test_loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images, targets = batch[0].to(self.device), batch[1].to(self.device)
                outputs = self.model(images)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        return 100. * correct / total
    
    def train(self, train_loader: DataLoader, test_loader: DataLoader, epochs: int):
        for epoch in range(epochs):
            train_acc = self.train_epoch(train_loader)
            val_acc = self.evaluate(test_loader)
            
            self.train_acc_trace.append(train_acc)
            self.val_acc_trace.append(val_acc)
            
            # if self.scheduler is not None:
            #     self.scheduler.step()
                
            print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")


def get_optimizers(model: nn.Module, opt_name, args):
    """Configure optimizers and schedulers."""
    total_steps = 50_000 // args.batch_size * args.epochs
    n_warmup = int(total_steps * 0.10)  # % of total steps
    weight_decay = 1e-4
    max_lr = 6e-4
    min_lr = 1e-6
    
    if opt_name == "Adam":
        # Adam
        adam = Adam(model.parameters(), lr=max_lr)
        adam_scheduler = WarmupCosineScheduler(
            adam, n_warmup, total_steps, min_lr, max_lr
        )
        optimizer = (adam, adam_scheduler, "Adam")
    
    elif opt_name == "AdamW":
        # AdamW
        adamw = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        adamw_scheduler = WarmupCosineScheduler(
            adamw, n_warmup, total_steps, min_lr, max_lr
        )
        optimizer = (adamw, adamw_scheduler, "AdamW")
    elif opt_name == "ADOPT":
        # ADOPT
        adopt = ADOPT(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        adopt_scheduler = WarmupCosineScheduler(
            adopt, n_warmup, total_steps, min_lr, max_lr
        )
        optimizer = (adopt, adopt_scheduler, "ADOPT")
    elif opt_name == "MARS":
        # MARS
        mars = MARS(model.parameters(), lr=3e-3, weight_decay=weight_decay, optimize_1d=False)
        mars_scheduler = WarmupCosineScheduler(
            mars, n_warmup, total_steps, min_lr, 3e-3
        )
        optimizer = (mars, mars_scheduler, "MARS")
    return optimizer


def plot_results(results: List[List[float]], optimizer_names: List[str], args):
    """Plot training results."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    colors = ["#74add1", "#1730bd", "#1a9850", "#001c01"]
    
    for i, acc in enumerate(results):
        ax.plot(range(1, len(acc) + 1), acc, label=optimizer_names[i], lw=2, color=colors[i])
    
    ax.set_title(f"{args.dataset.upper()} (val)", loc="left")
    ax.set_xlabel("Epoch", fontsize="medium")
    ax.set_ylabel("Accuracy (%)", fontsize="medium")
    
    ax.legend(ncols=2, columnspacing=0.8, fontsize="medium")
    ax.grid(alpha=0.2)
    
    ax.set_ylim(90 if args.dataset == "mnist" else 70)
    acc_min, acc_max = ax.get_ylim()
    ax.set_yticks(torch.linspace(acc_min, acc_max, 5).int().tolist())
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    fig.tight_layout()
    fig.savefig(
        f"./compare-{args.dataset}-blank.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def main(args):
    # Set random seed and device
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Get dataloaders
    train_loader, test_loader = get_datasets(args.dataset, args.batch_size)
    # Model configuration
    model_config = {
        "n_inputs": (3, 32, 32) if args.dataset == "cifar10" else (1, 28, 28),
        "conv_layers_list": [
            {"filters": 32, "kernel_size": 3, "repeat": 2, "batch_norm": True},
            {"filters": 64, "kernel_size": 3, "repeat": 2, "batch_norm": True},
            {"filters": 128, "kernel_size": 3, "repeat": 2, "batch_norm": True},
        ],
        "n_hiddens_list": [512],
        "n_outputs": 10,
        "dropout": 0.2,
    }
    
    results = []
    optimizer_names = []
    # Train with different optimizers
    opt_names = ["Adam", "AdamW", "ADOPT", "MARS"]
    for opt_name in opt_names:
        print(opt_name)
        torch.manual_seed(args.seed)
        model = Network(**model_config).to(device)
        optimizer, scheduler, name = get_optimizers(model, opt_name, args)
        trainer = Trainer(model, optimizer, scheduler, device)
        trainer.train(train_loader, test_loader, args.epochs)
        results.append(trainer.val_acc_trace)
        optimizer_names.append(name)
    
    plot_results(results, optimizer_names, args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)