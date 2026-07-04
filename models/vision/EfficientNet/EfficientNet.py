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
    
if __name__=="__main__":
    mbblock = MBConv(3, 100, 3, 1, 6, 0.25).to('cuda')
    sample = torch.randn([1,3,224,224]).to('cuda')
    outputs = mbblock(sample)
    print(outputs.shape)