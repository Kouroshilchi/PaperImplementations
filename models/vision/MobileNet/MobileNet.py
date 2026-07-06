import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        
        # Depthwise Convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,   
            groups=in_channels,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Pointwise Convolution
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,          
            padding=0,
            bias=False
        )
        
        self.bn2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise_conv(x)))
        x = self.relu(self.bn2(self.pointwise_conv(x)))
        return x