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
    
class ResblockSE(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride, padding, mode="basicblock", reduction_ratio=16):
        super().__init__()
        if mode == "basicblock":
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                SEBlock(out_channels, reduction_ratio=reduction_ratio)
            )
            if in_channels != out_channels or stride != 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Identity()
                
        elif mode == "bottleneckblock":
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
                SEBlock(out_channels * self.expansion, reduction_ratio=reduction_ratio)
            )
            if in_channels != out_channels * self.expansion or stride != 1:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels * self.expansion)
                )
            else:
                self.shortcut = nn.Identity()
    
    def forward(self, x):
        shortcut = x
        x = self.main(x)
        x = F.relu_(x + self.shortcut(shortcut))
        return x

class ResNet50SE(nn.Module):
    def __init__(self, in_channels, num_classes, r=16):
        super().__init__()
        self.in_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(64, 64, 3, r, stride=1)
        self.layer2 = self.make_layer(256, 128, 4, r, stride=2)
        self.layer3 = self.make_layer(512, 256, 6, r, stride=2)
        self.layer4 = self.make_layer(1024, 512, 3, r, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        
    def make_layer(self, in_channels, out_channels, num_blocks, r, stride=1):
        layers = []
        layers.append(ResblockSE(in_channels, out_channels, stride=stride, padding=1, 
                                mode="bottleneckblock", reduction_ratio=r))
        in_channels = out_channels * ResblockSE.expansion
        for i in range(num_blocks - 1):
            layers.append(ResblockSE(in_channels, out_channels, stride=1, padding=1,
                                    mode="bottleneckblock", reduction_ratio=r))
            in_channels = out_channels * ResblockSE.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.in_blocks(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

if __name__=="__main__":
    model = ResNet50SE(3, 100, 16).to('cuda')
    picture = torch.randn([1,3,224,224]).to('cuda')
    with torch.no_grad():
        outputs = model(picture)
    summary(model, (3,224,224))
    print(outputs.shape)