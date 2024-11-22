from typing import List, Tuple, Type, Union
import importlib

import torch
import torch.nn as nn


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")


class MLP(Base):
    """Multi-layer Perceptron."""
    def __init__(
        self,
        n_inputs: int,
        n_hiddens_list: Union[List, int],
        n_outputs: int,
        activation_f: str = "Tanh",
    ):
        super().__init__()
        
        if isinstance(n_hiddens_list, int):
            n_hiddens_list = [n_hiddens_list]
        
        if n_hiddens_list == [] or n_hiddens_list == [0]:
            self.n_hidden_layers = 0
        else:
            self.n_hidden_layers = len(n_hiddens_list)
        
        activation = get_activation(activation_f)
        
        layers = []
        ni = n_inputs
        if self.n_hidden_layers > 0:
            for _, n_units in enumerate(n_hiddens_list):
                layers.append(nn.Linear(ni, n_units))
                layers.append(activation())
                ni = n_units
        layers.append(nn.Linear(ni, n_outputs))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.layers(x)


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


if __name__ == "__main__":
    # Test MLP
    x = torch.randn(4, 10)
    model = MLP(n_inputs=x.shape[1], n_hiddens_list=[10, 10], n_outputs=2, activation_f="ReLU")
    model.summary()
    print("mlp output shape:", model(x).shape)
    
    # Test CNN
    x = torch.randn(4, 3, 32, 32)  # Note: PyTorch uses channels_first format
    model = Network(
        n_inputs=x.shape[1:],
        n_outputs=10,
        conv_layers_list=[{"filters": 8, "kernel_size": 3}, {"filters": 16, "kernel_size": 3}],
        n_hiddens_list=[32, 10],
        activation_f="GELU",
    )
    model.summary()
    print("cnn output shape:", model(x).shape)