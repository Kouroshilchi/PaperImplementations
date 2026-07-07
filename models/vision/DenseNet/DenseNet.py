import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
            )
            self.layers.append(layer)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)
    
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=1000):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.num_classes = num_classes

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 2 * growth_rate, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(2 * growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Dense blocks and transition layers
        num_channels = 2 * growth_rate
        self.blocks = nn.ModuleList()
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers, in_channels=num_channels, growth_rate=growth_rate)
            self.blocks.append(block)
            num_channels += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans_layer = TransitionLayer(in_channels=num_channels, out_channels=num_channels // 2)
                self.blocks.append(trans_layer)
                num_channels //= 2
        self.classifier = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        for block in self.blocks:
            x = block(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__=="__main__":
    model = DenseNet(32, num_classes=100).to('cuda')
    picture = torch.randn([1,3,224,224]).to('cuda')
    with torch.no_grad():
        outputs = model(picture)
    print(outputs.shape)
    summary(model, (3,224,224))