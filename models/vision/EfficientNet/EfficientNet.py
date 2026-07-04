import math
import torch
import torch.nn as nn
from torchsummary import summary


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor, se_ratio):
        super().__init__()
        hidden_channels = max(1, int(in_channels * expansion_factor))

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, max(1, int(hidden_channels * se_ratio))),
            nn.SiLU(inplace=True),
            nn.Linear(max(1, int(hidden_channels * se_ratio)), hidden_channels),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.use_residual = (stride == 1 and in_channels == out_channels)

    def forward(self, x):
        shortcut = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = x * self.se(x).unsqueeze(-1).unsqueeze(-1)
        x = self.project(x)
        if self.use_residual:
            x = x + shortcut
        return x


class EfficientNetBase(nn.Module):
    def __init__(self, width_coeff, depth_coeff, resolution, dropout_rate, in_channels=3, num_classes=1000):
        super().__init__()
        self.width_coeff = width_coeff
        self.depth_coeff = depth_coeff
        self.resolution = resolution
        self.dropout_rate = dropout_rate

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self._round_channels(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self._round_channels(32)),
            nn.SiLU(inplace=True),
        )

        self.features = self._build_features()
        self.head = nn.Sequential(
            nn.Conv2d(self._round_channels(320), self._round_channels(1280), kernel_size=1, bias=False),
            nn.BatchNorm2d(self._round_channels(1280)),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self._round_channels(1280), num_classes),
        )

    def _round_channels(self, channels):
        return max(1, int(channels * self.width_coeff))

    def _round_repeats(self, repeats):
        return max(1, int(math.ceil(repeats * self.depth_coeff)))

    def _build_features(self):
        stages = [
            (self._round_channels(32), 16, 3, 1, 1, 0.25, 1),
            (self._round_channels(16), self._round_channels(24), 3, 2, 6, 0.25, 2),
            (self._round_channels(24), self._round_channels(40), 5, 2, 6, 0.25, 2),
            (self._round_channels(40), self._round_channels(80), 3, 2, 6, 0.25, 3),
            (self._round_channels(80), self._round_channels(112), 5, 1, 6, 0.25, 3),
            (self._round_channels(112), self._round_channels(192), 5, 2, 6, 0.25, 4),
            (self._round_channels(192), self._round_channels(320), 3, 1, 6, 0.25, 1),
        ]

        layers = []
        in_channels = self._round_channels(32)
        for out_channels, next_channels, kernel_size, stride, expansion_factor, se_ratio, repeats in stages:
            layers.append(MBConv(in_channels, next_channels, kernel_size, stride, expansion_factor, se_ratio))
            in_channels = next_channels
            for _ in range(self._round_repeats(repeats) - 1):
                layers.append(MBConv(in_channels, next_channels, kernel_size, 1, expansion_factor, se_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x


class EfficientNetB0(EfficientNetBase):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__(width_coeff=1.0, depth_coeff=1.0, resolution=224, dropout_rate=0.2,
                         in_channels=in_channels, num_classes=num_classes)


class EfficientNetB1(EfficientNetBase):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__(width_coeff=1.0, depth_coeff=1.1, resolution=240, dropout_rate=0.2,
                         in_channels=in_channels, num_classes=num_classes)


class EfficientNetB2(EfficientNetBase):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__(width_coeff=1.1, depth_coeff=1.2, resolution=260, dropout_rate=0.3,
                         in_channels=in_channels, num_classes=num_classes)


class EfficientNetB3(EfficientNetBase):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__(width_coeff=1.2, depth_coeff=1.4, resolution=300, dropout_rate=0.3,
                         in_channels=in_channels, num_classes=num_classes)


class EfficientNetB4(EfficientNetBase):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__(width_coeff=1.4, depth_coeff=1.8, resolution=380, dropout_rate=0.4,
                         in_channels=in_channels, num_classes=num_classes)


class EfficientNetB5(EfficientNetBase):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__(width_coeff=1.6, depth_coeff=2.2, resolution=456, dropout_rate=0.4,
                         in_channels=in_channels, num_classes=num_classes)


class EfficientNetB6(EfficientNetBase):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__(width_coeff=1.8, depth_coeff=2.6, resolution=528, dropout_rate=0.5,
                         in_channels=in_channels, num_classes=num_classes)


class EfficientNetB7(EfficientNetBase):
    def __init__(self, in_channels=3, num_classes=1000):
        super().__init__(width_coeff=2.0, depth_coeff=3.1, resolution=600, dropout_rate=0.5,
                         in_channels=in_channels, num_classes=num_classes)


if __name__ == "__main__":
    model = EfficientNetB0(3, 100).to('cuda')
    sample = torch.randn([1, 3, 224, 224]).to('cuda')
    outputs = model(sample)
    summary(model, (3, 224, 224))
    print(outputs.shape)