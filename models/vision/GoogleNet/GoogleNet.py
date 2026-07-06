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

if __name__=="__main__":
    block = InceptionBlock(3, 64, 96, 128, 16, 32, 32).to('cuda')
    picture = torch.randn([1,3,224,224]).to('cuda')
    with torch.no_grad():
        outputs = block(picture)
    print (outputs.shape)