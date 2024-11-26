import numpy as np
import torch

import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_model(args):
    """
    models including:
    - VGG16
    - resnet18
    
    """
    if args.net == 'simple_cnn':
        from .model_CNN import Network
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
        model = Network(**model_config)
    else:
        try:
            model = torchvision.models.get_model(args.net, num_classes=10)
        except:
            print('Model not found')
            raise NotImplementedError
    return model

def get_datasets(dataset_name: str, train_batch_size: int, eval_batch_size: int):
    """Get train and test dataloaders."""
    print('==> Preparing data..')
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    elif dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    else:
        raise NotImplementedError(f"{dataset_name=} is not implemented.")

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=4)
    
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
    def get_lr(self):
        lr_list = []
        for param_group in self.optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list

class ConstantScheduler:
    """Constant learning rate scheduler."""
    def __init__(self, optimizer, lr: float):
        self.optimizer = optimizer
        self.lr = lr
        
    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def get_lr(self):
        lr_list = []
        for param_group in self.optimizer.param_groups:
            lr_list.append(param_group['lr'])
        return lr_list
            
def get_scheduler(optimizer, args):   
    if args.scheduler == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        scheduler = MultiStepLR(optimizer, milestones=[args.Nepoch // 2, (args.Nepoch * 3) // 4], gamma=0.1) 
    elif args.scheduler == 'cosine':
        scheduler = WarmupCosineScheduler(optimizer, warmup_iters = args.Nepoch // 10, total_iters = args.Nepoch, 
                                          min_lr = 0., max_lr = args.lr)
    elif args.scheduler == 'constant':
        scheduler = ConstantScheduler(optimizer, lr = args.lr)
    return scheduler