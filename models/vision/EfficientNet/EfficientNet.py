import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor, se_ratio):
        super().__init__()
        main_channels_size = in_channels*expansion_factor

        self.silu = nn.SiLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(in_channels, main_channels_size,kernel_size=(1,1))
        self.conv2 = nn.Conv2d(main_channels_size, main_channels_size, kernel_size, stride, padding=kernel_size//2, groups=main_channels_size)
        self.conv3 = nn.Conv2d(main_channels_size, out_channels, kernel_size=(1,1), stride=1)
        
        self.bn1 = nn.BatchNorm2d(main_channels_size)
        self.bn2 = nn.BatchNorm2d(main_channels_size)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.fc1 = nn.Linear(main_channels_size, int(main_channels_size*se_ratio))
        self.fc2 = nn.Linear(int(main_channels_size*se_ratio), main_channels_size)

        self.use_residual = (stride == 1 and in_channels == out_channels)
        
    def forward(self, x):
        shortcut = x
        x = self.silu(self.bn1(self.conv1(x)))
        x = self.silu(self.bn2(self.conv2(x)))

        x_ = self.global_avgpool(x).squeeze(-1).squeeze(-1)
        x_ = self.silu(self.fc1(x_))
        x_ = self.sigmoid(self.fc2(x_)).unsqueeze(-1).unsqueeze(-1)

        x = x * x_

        x = self.bn3(self.conv3(x))
        if self.use_residual:
            x += shortcut
        return x
    
class EfficientNetB0(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True)
        )
        
        # Body (Main stages)
        self.main = nn.Sequential(
            # Stage 2: 16 channels
            MBConv(32,  16, kernel_size=3, stride=1, expansion_factor=1,  se_ratio=0.25),
            
            # Stage 3: 24 channels
            MBConv(16,  24, kernel_size=3, stride=2, expansion_factor=6,  se_ratio=0.25),
            MBConv(24,  24, kernel_size=3, stride=1, expansion_factor=6,  se_ratio=0.25),
            
            # Stage 4: 40 channels
            MBConv(24,  40, kernel_size=5, stride=2, expansion_factor=6,  se_ratio=0.25),
            MBConv(40,  40, kernel_size=5, stride=1, expansion_factor=6,  se_ratio=0.25),
            
            # Stage 5: 80 channels
            MBConv(40,  80, kernel_size=3, stride=2, expansion_factor=6,  se_ratio=0.25),
            MBConv(80,  80, kernel_size=3, stride=1, expansion_factor=6,  se_ratio=0.25),
            MBConv(80,  80, kernel_size=3, stride=1, expansion_factor=6,  se_ratio=0.25),
            
            # Stage 6: 112 channels
            MBConv(80, 112, kernel_size=5, stride=1, expansion_factor=6,  se_ratio=0.25),
            MBConv(112,112, kernel_size=5, stride=1, expansion_factor=6,  se_ratio=0.25),
            MBConv(112,112, kernel_size=5, stride=1, expansion_factor=6,  se_ratio=0.25),
            
            # Stage 7: 192 channels
            MBConv(112,192, kernel_size=5, stride=2, expansion_factor=6,  se_ratio=0.25),
            MBConv(192,192, kernel_size=5, stride=1, expansion_factor=6,  se_ratio=0.25),
            MBConv(192,192, kernel_size=5, stride=1, expansion_factor=6,  se_ratio=0.25),
            MBConv(192,192, kernel_size=5, stride=1, expansion_factor=6,  se_ratio=0.25),
            
            # Stage 8: 320 channels
            MBConv(192,320, kernel_size=3, stride=1, expansion_factor=6,  se_ratio=0.25),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes)
        )
        
    def forward(self, x):
        x = self.stem(x)
        x = self.main(x)
        x = self.head(x)
        return x
        


if __name__=="__main__":
    model = EfficientNetB0(3, 100).to('cuda')
    sample = torch.randn([1,3,224,224]).to('cuda')
    outputs = model(sample)
    summary(model, (3,224,224))
    print(outputs.shape)