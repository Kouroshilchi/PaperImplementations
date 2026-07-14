import torch
import torch.nn as nn
import torch.nn.functional as F
from  torchsummary import summary 

class ConvNeXtBlock(nn.Module):
    expansion = 4
    def __init__(self, dim, layer_scale_init=1e-6):
        super().__init__() 
        self.dim = dim
        self.depthwise_conv = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=7,
            stride=1,
            padding=3,   
            groups=dim
        )
        self.ln = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, dim * self.expansion)
        self.gelu = nn.GELU()
        self.pwconv2 = nn.Linear(dim * self.expansion, dim)
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones((1,1,1,dim)),
            requires_grad=True
        ) if layer_scale_init > 0 else None
    def forward(self, x):
        x_ = x
        x = self.depthwise_conv(x).permute(0,2,3,1)
        x = self.ln(x)
        x = self.pwconv1(x)
        x = self.gelu(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma*x
        x = x.permute(0,3,1,2) + x_
        return x
    
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )
        self.layernorm = nn.LayerNorm(in_channels)
    def forward(self, x):
        x = self.layernorm(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = self.downsample(x)
        return x
    
class LayerNorm_withpermut(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.ln(x.permute(0,2,3,1)).permute(0,3,1,2)
        return x
    
class ConvNeXt(nn.Module):
    def __init__(self, in_channels, num_classes, depths=[3,3,9,3], dims=[96,192,384,768]):
        super().__init__()
        
        self.in_blocks = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=dims[0], kernel_size=4, stride=4),
            LayerNorm_withpermut(dims[0])
        )
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            self.downsample_layers.append(
                Downsample(in_channels=dims[i], out_channels=dims[i+1])
            )
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        self.final_layernorm = LayerNorm_withpermut(dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dims[-1], num_classes)
    def forward(self, x):
        x = self.in_blocks(x)
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downsample_layers[i](x)
        x = self.final_layernorm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = ConvNeXt(in_channels=3, num_classes=100).to('cuda')
    summary(model, (3, 224, 224))