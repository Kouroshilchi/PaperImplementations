import torch 
import torch.nn as nn
import torch.nn.functional as F

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels,kernel_size=3, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )


        if in_channels != out_channels or stride!= 1:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        shortcut = x
        x = self.main(x)
        x = F.relu(x + self.shortcut(shortcut))
        return x