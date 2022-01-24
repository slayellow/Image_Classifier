import os
import math
import torch
import torch.nn as nn


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class ResNetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.se_module = SEModule(planes, reduction=16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)
        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.se_module = SEModule(planes * 4, reduction=16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class ResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride, padding=1, groups=32, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 2)
        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.se_module = SEModule(planes * 4, reduction=16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SENet(nn.Module):
    # SE-ResNet-18     :    SE-ResNeXt-18   :   BasicBlock        :   [2, 2, 2, 2]
    # SE-ResNet-34     :    SE-ResNeXt-18   :   BasicBlock        :   [3, 4, 6, 3]
    # SE-ResNet-50     :    SE-ResNeXt-18   :   BottleNeckBlock   :   [3, 4, 6, 3]
    # SE-ResNet-101    :    SE-ResNeXt-18   :   BottleNeckBlock   :   [3, 4, 23, 3]
    # SE-ResNet-152    :    SE-ResNeXt-18   :   BottleNeckBlock   :   [3, 4, 36, 3]
    # Channel       : [64, 128, 256, 512]
    def __init__(self, layer_num, num_classes=1000, resnext=False):
        super(SENet, self).__init__()

        if resnext:
            self.model_name = 'SE_ResNeXt_{}'.format(layer_num)
        else:
            self.model_name = 'SE_ResNet_{}'.format(layer_num)
        # SENet의 기본 구성
        blocks = {18: (2, 2, 2, 2),
                  34: (3, 4, 6, 3),
                  50: (3, 4, 6, 3),
                  101: (3, 4, 23, 3),
                  152: (3, 4, 36, 3)}
        channels = (64, 128, 256, 512)
        in_channel = 3
        self.inplanes = 64

        if layer_num == 18 or layer_num == 34:
            block = ResNetBasicBlock
        elif layer_num == 50 or layer_num == 101 or layer_num == 152:
            if resnext:
                block = ResNeXtBottleneck
            else:
                block = ResNetBottleneck
        else:
            print("Not Correct Model Layer Number")

        self.conv1 = nn.Conv2d(in_channel, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block
        self.layer1 = self._make_layer(block, channels[0], blocks[layer_num][0])
        self.layer2 = self._make_layer(block, channels[1], blocks[layer_num][1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], blocks[layer_num][2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], blocks[layer_num][3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(channels[3] * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def senet(layer_num, classes, resnext, pretrained=False, pretrained_path=None):
    model = SENet(layer_num=layer_num, num_classes=classes, resnext=resnext)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    return model

