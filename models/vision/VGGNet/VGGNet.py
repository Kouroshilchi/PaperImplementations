import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,  lrn=False, num_layers=1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride, padding="same"))
            if lrn and i==num_layers-1:
                layers.append(nn.LocalResponseNorm(out_channels, alpha=0.0001, beta=0.75, k=2))
            else:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
