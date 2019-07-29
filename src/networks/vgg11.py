from src.networks.networks import NetworkBase
import torch.nn as nn
import torch

class VGG11(NetworkBase):
    def __init__(self, num_classes=10):
        super(VGG11, self).__init__()
        config_rgb = [64, 'M', 128, 'M', 256, 256]
        config_depth = [64, 'M', 128, 'M', 256, 256]
        config_feat = ['M', 512, 512, 'M', 512, 512, 'M']

        self._enc_rgb = self._make_layers(config_rgb, 3)
        self._enc_depth = self._make_layers(config_depth, 1)
        self._features = self._make_layers(config_feat, 512)
        self._classifier = nn.Linear(512, num_classes)

        # TODO this is just an example, this should load pytorch pretrained weights instead random
        self.init_weights(self)

    def forward(self, rgb, depth):
        enc_rgb = self._enc_rgb(rgb)
        enc_depth = self._enc_depth(depth)
        out = self._features(torch.cat([enc_rgb, enc_depth], -3))
        out = out.view(out.size(0), -1)
        return self._classifier(out)

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)
