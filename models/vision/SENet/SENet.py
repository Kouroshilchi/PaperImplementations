import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super().__init__()
        self.globalpool = nn.AdaptiveAvgPool2d((1,1))
        self.bottleneck = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        weights = self.bottleneck(self.globalpool(x).view(x.size(0), -1)).unsqueeze(-1).unsqueeze(-1)
        x = x * weights
        return x


if __name__=="__main__":
    model = SEBlock(64, 16)
    picture = torch.randn([1,64,224,224])
    with torch.no_grad():
        outputs = model(picture)
    print(outputs.shape)