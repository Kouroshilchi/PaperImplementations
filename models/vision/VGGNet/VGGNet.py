import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,  lrn=False, num_layers=1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride, padding="same"))
            if lrn and i==num_layers-1:
                layers.append(nn.LocalResponseNorm(out_channels, alpha=0.0001, beta=0.75, k=2))
            else:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)


class VGG16(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.layers = nn.Sequential(

            ConvBlock(in_channels,64,3,1,True,2),
            nn.MaxPool2d(2, 2), # 224 -> 112

            ConvBlock(64,128,3,1,True,2),
            nn.MaxPool2d(2, 2), # 112 -> 56

            ConvBlock(128,256,3,1,True,3),
            nn.MaxPool2d(2, 2), # 56 -> 28

            ConvBlock(256,512,3,1,True,3),
            nn.MaxPool2d(2, 2), # 28 -> 14

            ConvBlock(512,512,3,1,True,3),
            nn.MaxPool2d(2, 2), # 14 -> 7

            nn.Flatten(),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
            
        )
    def forward(self, x):
        return self.layers(x)


if __name__=="__main__":
    model = VGG16(3,1000).to('cuda')
    picture = torch.randn([1,3,224,224]).to('cuda')
    with torch.no_grad():
        outputs = model(picture)
    print(outputs.shape)
    summary(model, (3,224,224))