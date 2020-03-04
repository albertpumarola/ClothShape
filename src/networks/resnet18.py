from src.networks.networks import NetworkBase
import torch.nn as nn
import torch.nn.functional as F
import torch

class ResNet18(NetworkBase):
    def __init__(self, num_classes=10, use_depth=False):
        super(ResNet18, self).__init__()
        self._name = "ResNet18"
        self._in_planes = 64
        self._use_depth = use_depth

        self._conv1_rgb = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn1_rgb = nn.BatchNorm2d(64)
        self._layer1_rgb = self._make_layer(ResNetBlock, [64, 64], [64, 64], 2, stride=1)
        self._layer2_rgb = self._make_layer(ResNetBlock, [64, 128], [128, 128], 2, stride=2)

        if use_depth:
            self._conv1_depth = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self._bn1_depth = nn.BatchNorm2d(64)
            self._layer1_depth = self._make_layer(ResNetBlock, [64, 64], [64, 64], 2, stride=1)
            self._layer2_depth = self._make_layer(ResNetBlock, [64, 128], [128, 128], 2, stride=2)

        layer3_nc = 256 if use_depth else 128
        self._layer3 = self._make_layer(ResNetBlock, [layer3_nc, layer3_nc], [256, 256], 2, stride=2)
        self._layer4 = self._make_layer(ResNetBlock, [256, 512], [512, 512], 2, stride=2)
        self._linear = nn.Linear(512 * ResNetBlock.expansion, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.init_weights(self)



    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for in_plane, out_plane, stride in zip(in_planes, out_planes, strides):
            layers.append(block(in_plane, out_plane, stride))
        return nn.Sequential(*layers)

    def forward(self, rgb, depth=None):
        enc_rgb = F.relu(self._bn1_rgb(self._conv1_rgb(rgb)))
        enc_rgb = self._layer1_rgb(enc_rgb)
        enc_rgb = self._layer2_rgb(enc_rgb)

        if self._use_depth:
            enc_depth = F.relu(self._bn1_depth(self._conv1_depth(depth)))
            enc_depth = self._layer1_rgb(enc_depth)
            enc_depth = self._layer2_rgb(enc_depth)
            features = torch.cat([enc_rgb, enc_depth], -3)
            
        else:
            features = enc_rgb

        out = self._layer3(features)
        out = self._layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self._linear(out)
        return out


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ResNetBlock, self).__init__()
        self._conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self._bn1 = nn.BatchNorm2d(planes)
        self._conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self._bn2 = nn.BatchNorm2d(planes)

        self._shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self._shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self._bn1(self._conv1(x)))
        out = self._bn2(self._conv2(out))
        out += self._shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self._conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(planes)
        self._conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self._bn2 = nn.BatchNorm2d(planes)
        self._conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self._bn3 = nn.BatchNorm2d(self.expansion*planes)

        self._shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self._shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self._bn1(self._conv1(x)))
        out = F.relu(self._bn2(self._conv2(out)))
        out = self._bn3(self._conv3(out))
        out += self._shortcut(x)
        out = F.relu(out)
        return out
