import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
# in_channels , out_channels, stride, num_layers
mobile_net_architecture = [
    (32, 64, 1, 1),   # Stage 1
    (64, 128, 2, 1),  # Stage 2
    (128, 128, 1, 1), # Stage 3
    (128, 256, 2, 1), # Stage 4
    (256, 256, 1, 1), # Stage 5
    (256, 512, 2, 1), # Stage 6
    (512, 512, 1, 5), # Stage 7 
    (512, 1024, 2, 1),# Stage 8
    (1024, 1024, 1, 1)# Stage 9
]

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
        layers = []
        for in_ch, out_ch, stride, num_layers in mobile_net_architecture:
            for i in range(num_layers):
                layers.append(DepthwiseSeparableBlock(self.get_ch(in_ch), self.get_ch(out_ch), stride=stride if i==0 else 1))
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.get_ch(1024), num_classes)
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