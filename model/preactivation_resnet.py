import math
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)

        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        if self.downsample is not None:
            residual = self.downsample(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual

        return out


class PreActivationResNet(nn.Module):
    # pre-activation ResNet-18     : BasicBlock        :   [2, 2, 2, 2]
    # pre-activation ResNet-34     : BasicBlock        :   [3, 4, 6, 3]
    # pre-activation ResNet-50     : BottleNeckBlock   :   [3, 4, 6, 3]
    # pre-activation ResNet-101    : BottleNeckBlock   :   [3, 4, 23, 3]
    # pre-activation ResNet-152    : BottleNeckBlock   :   [3, 4, 36, 3]
    # Channel       : [64, 128, 256, 512]
    def __init__(self, layer_num, num_classes=1000):
        super(PreActivationResNet, self).__init__()

        self.model_name = 'Pre-Activation_ResNet_{}'.format(layer_num)

        # ResNet??? ?????? ??????
        blocks = {18: (2, 2, 2, 2),
                  34: (3, 4, 6, 3),
                  50: (3, 4, 6, 3),
                  101: (3, 4, 23, 3),
                  152: (3, 4, 36, 3)}
        channels = (64, 128, 256, 512)
        in_channel = 3
        self.inplanes = 64

        if layer_num == 18 or layer_num == 34:
            block = BasicBlock
        elif layer_num == 50 or layer_num == 101 or layer_num == 152:
            block = Bottleneck
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
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False))
        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # First Layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual Block
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def preactivation_resnet(layer_num, classes, pretrained=False, pretrained_path=None):
    model = PreActivationResNet(layer_num=layer_num, num_classes=classes)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    return model
