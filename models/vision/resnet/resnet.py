import torch 
import torch.nn as nn
import torch.nn.functional as F

class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding, mode="basicblock"):
        super().__init__()
        if mode=="basicblock":
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,kernel_size=3, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,kernel_size=3, stride=1, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif mode == "bottleneckblock":
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels // 4,kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels // 4,kernel_size=3, stride=1, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels,kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels),
            )


        if in_channels != out_channels or stride!= 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        shortcut = x
        x = self.main(x)
        x = F.relu_(x + self.shortcut(shortcut))
        return x


class ResNet50(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_blocks = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(64, 256, 3)
        self.layer2 = self.make_layer(256, 512, 4, stride=2)
        self.layer3 = self.make_layer(512, 1024, 6, stride=2)
        self.layer4 = self.make_layer(1024, 2048, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(...)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(Resblock(in_channels, out_channels, stride=stride, padding=1, mode="bottleneckblock"))
        for i in range(num_blocks-1):
            layers.append(Resblock(out_channels, out_channels, stride=1, padding=1, mode="bottleneckblock"))
        return nn.Sequential(*layers)
        
    

    def forward(self, x):
        x = self.in_blocks(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__=="__main__":
    model = ResNet50(3, 1000).to('cuda')
    pictures = torch.randn([1,3,224,224] , device='cuda')
    outputs = model(pictures)
    print(outputs.shape)