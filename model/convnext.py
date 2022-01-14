import torch
import torch.nn as nn
import os
from model.utils.droppath import DropPath
import math
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channel, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=7, padding=3, groups=in_channel, bias=False)
        self.norm = LayerNorm(in_channel, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channel, 4 * in_channel)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channel, in_channel)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channel)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXt(nn.Module):
    # ConvNeXt-T     : [3, 3, 9, 3]     Channel : [96, 192, 384, 768]
    # ConvNeXt-S     : [3, 3, 27, 3]    Channel : [96, 192, 384, 768]
    # ConvNeXt-B     : [3, 3, 27, 3]    Channel : [128, 256, 512, 1024]
    # ConvNeXt-L     : [3, 3, 27, 3]    Channel : [192, 384, 768, 1536]
    # ConvNeXt-XL    : [3, 3, 27, 3]    Channel : [256, 512, 1024, 2048]
    def __init__(self, version, num_classes=1000):
        super(ConvNeXt, self).__init__()

        self.model_name = 'ConvNeXt_{}'.format(version)

        # ResNet의 기본 구성
        blocks = {'T': (3, 3, 9, 3),
                  'S': (3, 3, 27, 3),
                  'B': (3, 3, 27, 3),
                  'L': (3, 3, 27, 3),
                  'XL': (3, 3, 27, 3)}

        channels = {'T': (96, 192, 384, 768),
                    'S': (96, 192, 384, 768),
                    'B': (128, 256, 512, 1024),
                    'L': (192, 384, 768, 1536),
                    'XL': (256, 512, 1024, 2048)}
        drop_path_rate = {'T': 0.1,
                          'S': 0.4,
                          'B': 0.5,
                          'L': 0.5,
                          'XL': 0.5}

        in_channel = 3

        self.downsample_layers = nn.ModuleList()
        # stem Stage
        stem = nn.Sequential(nn.Conv2d(in_channel, channels[version][0], kernel_size=4, stride=4, bias=False),
                             LayerNorm(channels[version][0], eps=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)

        # Downsampling Stage Between stem, res2, res3, res4, res5
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(channels[version][i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(channels[version][i], channels[version][i + 1], kernel_size=2, stride=2, bias=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = drop_path_rate[version]
        for i in range(4):
            stage = nn.Sequential(
                *[Block(channels[version][i], drop_path=dp_rates,
                        layer_scale_init_value=1e-6) for j in range(blocks[version][i])]
            )
            self.stages.append(stage)

        # Final Stage
        self.norm = nn.LayerNorm(channels[version][-1], eps=1e-6)
        self.head = nn.Linear(channels[version][-1], num_classes)

        # Initialize Weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_name(self):
        return self.model_name

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def convnext(version, classes, pretrained=False, pretrained_path=None):
    model = ConvNeXt(version=version, num_classes=classes)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    return model
