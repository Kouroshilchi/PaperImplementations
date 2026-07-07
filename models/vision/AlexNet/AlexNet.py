import torch 
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.lrn1 = nn.LocalResponseNorm(96, alpha=0.0001, beta=0.75, k=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.lrn2 = nn.LocalResponseNorm(256, alpha=0.0001, beta=0.75, k=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
    def forward(self, x):
        x = self.lrn1(self.maxpool(self.relu(self.conv1(x))))
        x = self.lrn2(self.maxpool(self.relu(self.conv2(x))))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.maxpool(self.relu(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__=="__main__":
    model = AlexNet(3, 100).to('cuda')
    picture = torch.randn([1,3,224,224]).to('cuda')
    with torch.no_grad():
        outputs = model(picture)
    print(outputs.shape)
    summary(model , input_size=(3,224,224))
        