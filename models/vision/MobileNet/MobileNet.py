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
    
class MobileNet(nn.Module):
    def __init__(self, in_channels, num_classes, a):
        super().__init__()
        self.a = a
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, self.get_ch(32), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.get_ch(32)),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            DepthwiseSeparableBlock(self.get_ch(32),  self.get_ch(64),  stride=1),
            
            DepthwiseSeparableBlock(self.get_ch(64),  self.get_ch(128), stride=2),
            DepthwiseSeparableBlock(self.get_ch(128), self.get_ch(128), stride=1),
            
            DepthwiseSeparableBlock(self.get_ch(128), self.get_ch(256), stride=2),
            DepthwiseSeparableBlock(self.get_ch(256), self.get_ch(256), stride=1),
            
            DepthwiseSeparableBlock(self.get_ch(256), self.get_ch(512), stride=2),
            
            DepthwiseSeparableBlock(self.get_ch(512), self.get_ch(512), stride=1),
            DepthwiseSeparableBlock(self.get_ch(512), self.get_ch(512), stride=1),
            DepthwiseSeparableBlock(self.get_ch(512), self.get_ch(512), stride=1),
            DepthwiseSeparableBlock(self.get_ch(512), self.get_ch(512), stride=1),
            DepthwiseSeparableBlock(self.get_ch(512), self.get_ch(512), stride=1),
            
            DepthwiseSeparableBlock(self.get_ch(512), self.get_ch(1024), stride=2),
            DepthwiseSeparableBlock(self.get_ch(1024), self.get_ch(1024), stride=1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )        

    def get_ch(self, num_channels):
        return int(num_channels*self.a)
    
    def forward(self, x):
        x = self.bottleneck(x)
        x = self.body(x)
        x = self.head(x)
        return x

if __name__=="__main__":
    model = MobileNet(3, 100, 1).to('cuda')
    picture = torch.randn([1,3,224,224]).to('cuda')
    with torch.no_grad():
        outputs = model(picture)
    print(outputs.shape)
    summary(model, (3,224,224))