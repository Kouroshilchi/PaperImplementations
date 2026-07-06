import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()
        self.a_branch = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channels)   -> batchnorm was not used in Inceptionv1
        )
        self.b_branch = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channels),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channels)
        )
        self.c_branch = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channels),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channels)
        )
        self.d_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(out_channels),
        )
        
    def forward(self, x):
        x_a = self.a_branch(x)
        x_b = self.b_branch(x)
        x_c = self.c_branch(x)
        x_d = self.d_branch(x)
        x = torch.cat([x_a, x_b, x_c, x_d], dim=1)
        return x
    

class GoogleNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True), 
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.layers = nn.Sequential(
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),   # 3a
            InceptionBlock(256, 128, 128, 192, 32, 96, 64), # 3b
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),   # 4a
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),  # 4b
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),  # 4c
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),  # 4d
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),# 4e
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            InceptionBlock(832, 256, 160, 320, 32, 128, 128),# 5a
            InceptionBlock(832, 384, 192, 384, 48, 128, 128),# 5b
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024,num_classes),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.bottleneck(x)
        x = self.layers(x)
        x = self.head(x)
        return x



if __name__=="__main__":
    block = GoogleNet(3, 100).to('cuda')
    picture = torch.randn([1,3,224,224]).to('cuda')
    with torch.no_grad():
        outputs = block(picture)
    summary(block, (3,224,224))
    print(outputs.shape)