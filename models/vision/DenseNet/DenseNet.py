import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_channel, theta):
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, int(in_channel*theta), kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        x = self.layers(x)
        return x
    
class DenseNet(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=1000, 
                 growth_rate=32,      
                 theta=0.5,
                 architecture=(6, 12, 24, 16)): 
        
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.blocks = nn.ModuleList()
        in_channels = 64
        
        for i, num_layers in enumerate(architecture):
            self.blocks.append(DenseBlock(num_layers, in_channels, growth_rate))
            
            in_channels = in_channels + num_layers * growth_rate
            
            if i != len(architecture) - 1:
                self.blocks.append(TransitionBlock(in_channels, theta))
                in_channels = int(in_channels * theta)
        
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        for block in self.blocks:
            x = block(x)
        x = self.classifier(x)
        return x

if __name__=="__main__":
    model = DenseNet(3, 100).to('cuda')
    picture = torch.randn([1, 3, 224, 224]).to('cuda')
    with torch.no_grad():
        outputs = model(picture)
    summary(model, (3, 224, 224))
    print(outputs.shape)