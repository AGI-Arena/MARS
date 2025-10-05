from typing import List, Tuple, Type, Union
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_activation(activation_f: str) -> Type:
    """Get PyTorch activation function by name."""
    package_name = "torch.nn"
    module = importlib.import_module(package_name)
    
    activations = [getattr(module, attr) for attr in dir(module)]
    activations = [
        cls for cls in activations if isinstance(cls, type) and issubclass(cls, nn.Module)
    ]
    names = [cls.__name__.lower() for cls in activations]
    
    try:
        index = names.index(activation_f.lower())
        return activations[index]
    except ValueError:
        raise NotImplementedError(f"get_activation: {activation_f=} is not yet implemented.")


def compute_padding(
    input_size: tuple, kernel_size: int | tuple, stride: int | tuple = 1, dilation: int | tuple = 1
) -> Tuple[int, int]:
    """Compute padding for 'same' convolution."""
    if len(input_size) == 2:
        input_size = (*input_size, 1)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    
    input_h, input_w, _ = input_size
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation
    
    # Compute the effective kernel size after dilation
    effective_kernel_h = (kernel_h - 1) * dilation_h + 1
    effective_kernel_w = (kernel_w - 1) * dilation_w + 1
    
    # Compute the padding needed for same convolution
    pad_h = int(max((input_h - 1) * stride_h + effective_kernel_h - input_h, 0))
    pad_w = int(max((input_w - 1) * stride_w + effective_kernel_w - input_w, 0))
    
    # Compute the padding for each side
    pad_top = pad_h // 2
    pad_left = pad_w // 2
    
    return (pad_top, pad_left)


class Base(nn.Module):
    """Base class for neural network models."""
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    @property
    def shapes(self):
        return {name: p.shape for name, p in self.named_parameters()}
    
    def summary(self):
        print(self)
        print(f"Number of parameters: {self.num_params}")


class Network(Base):
    """Fully Connected / Convolutional Neural Network
    
    Args:
        n_inputs (Union[List[int], Tuple[int], torch.Size]): Input shape
        n_outputs (int): Number of output classes
        conv_layers_list (List[dict], optional): List of convolutional layers. Defaults to [].
        n_hiddens_list (Union[List, int], optional): List of hidden units. Defaults to 0.
        activation_f (str, optional): Activation function. Defaults to "ReLU".
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    
    conv_layers_list dict keys:
        filters: int
        kernel_size: int
        stride: int
        dilation: int
        padding: int
        bias: bool
        batch_norm: bool
        repeat: int
    """
    def __init__(
        self,
        n_inputs: Union[List[int], Tuple[int], torch.Size],
        n_outputs: int,
        conv_layers_list: List[dict] = [],
        n_hiddens_list: Union[List, int] = 0,
        activation_f: str = "ReLU",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if isinstance(n_hiddens_list, int):
            n_hiddens_list = [n_hiddens_list]
        
        if n_hiddens_list == [] or n_hiddens_list == [0]:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)
        
        activation = get_activation(activation_f)
        
        # Convert n_inputs to tensor for shape calculations
        ni = torch.tensor(n_inputs)
        
        conv_layers = []
        if conv_layers_list:
            for conv_layer in conv_layers_list:
                n_channels = int(ni[0])
                
                padding = conv_layer.get(
                    "padding",
                    compute_padding(  # same padding
                        tuple(ni.tolist()),
                        conv_layer["kernel_size"],
                        conv_layer.get("stride", 1),
                        conv_layer.get("dilation", 1),
                    ),
                )
                
                # Add repeated conv blocks
                for i in range(conv_layer.get("repeat", 1)):
                    # Convolutional layer
                    conv_layers.append(
                        nn.Conv2d(
                            n_channels if i == 0 else conv_layer["filters"],
                            conv_layer["filters"],
                            conv_layer["kernel_size"],
                            stride=conv_layer.get("stride", 1),
                            padding=padding,
                            dilation=conv_layer.get("dilation", 1),
                            bias=conv_layer.get("bias", True),
                        )
                    )
                    
                    # Activation
                    conv_layers.append(activation())
                    
                    # Optional batch norm
                    if conv_layer.get("batch_norm"):
                        conv_layers.append(nn.BatchNorm2d(conv_layer["filters"]))
                
                # Max pooling after each conv block
                conv_layers.append(nn.MaxPool2d(2, stride=2))
                
                # Optional dropout
                if dropout > 0:
                    conv_layers.append(nn.Dropout(dropout))
                
                # Update input shape for next layer
                ni = torch.cat([torch.tensor([conv_layer["filters"]]), ni[1:] // 2])
        
        self.conv = nn.Sequential(*conv_layers)
        
        # Fully connected layers
        ni = int(torch.prod(ni))
        fcn_layers = []
        if self.n_hidden_layers > 0:
            for _, n_units in enumerate(n_hiddens_list):
                fcn_layers.extend([
                    nn.Linear(ni, n_units),
                    activation()
                ])
                if dropout > 0:
                    fcn_layers.append(nn.Dropout(dropout))
                ni = n_units
        
        self.fcn = nn.Sequential(*fcn_layers)
        self.output = nn.Linear(ni, n_outputs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fcn(x)
        return self.output(x)
    
'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes = 10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes = num_classes)

def ResNet34(num_classes = 10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes = num_classes)

def ResNet50(num_classes = 10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes = num_classes)

def ResNet101(num_classes = 10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes = num_classes)

def ResNet152(num_classes = 10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes = num_classes)