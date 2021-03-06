import math
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class RoR(nn.Module):
    def __init__(self, layer_num, num_classes=1000):
        super(RoR, self).__init__()

        self.model_name = 'RoR3_{}'.format(layer_num)

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

        self.conv0_level1_shortcut = nn.Conv2d(channels[0], channels[3] * block.expansion, kernel_size=1, stride=8,
                                               bias=False)

        # Block
        self.layer1_level2_shortcut = nn.Conv2d(self.inplanes, channels[0] * block.expansion, kernel_size=1, bias=False)
        self.layer1 = self._make_layer(block, channels[0], blocks[layer_num][0])
        self.layer2_level2_shortcut = nn.Conv2d(self.inplanes, channels[1] * block.expansion, kernel_size=1, stride=2,
                                                bias=False)
        self.layer2 = self._make_layer(block, channels[1], blocks[layer_num][1], stride=2)
        self.layer3_level2_shortcut = nn.Conv2d(self.inplanes, channels[2] * block.expansion, kernel_size=1, stride=2,
                                                bias=False)
        self.layer3 = self._make_layer(block, channels[2], blocks[layer_num][2], stride=2)
        self.layer4_level2_shortcut = nn.Conv2d(self.inplanes, channels[3] * block.expansion, kernel_size=1, stride=2,
                                                bias=False)
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

        layer1_shortcut = self.conv0_level1_shortcut(x)

        layer1_level2_shortcut = self.layer1_level2_shortcut(x)
        x = self.layer1(x)
        x += layer1_level2_shortcut
        x = self.relu(x)

        layer2_level2_shortcut = self.layer2_level2_shortcut(x)
        x = self.layer2(x)
        x += layer2_level2_shortcut
        x = self.relu(x)

        layer3_level2_shortcut = self.layer3_level2_shortcut(x)
        x = self.layer3(x)
        x += layer3_level2_shortcut
        x = self.relu(x)

        layer4_level2_shortcut = self.layer4_level2_shortcut(x)
        x = self.layer4(x)
        x += layer4_level2_shortcut
        x += layer1_shortcut
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ror(layer_num, classes, pretrained=False, pretrained_path=None):
    model = RoR(layer_num=layer_num, num_classes=classes)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    return model
