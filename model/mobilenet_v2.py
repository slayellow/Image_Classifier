import math
import torch
import torch.nn as nn


class InvertedBottleneck(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, expansion=1):
        super(InvertedBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel * expansion, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel * expansion)

        self.conv2 = nn.Conv2d(in_channel * expansion, in_channel * expansion, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=in_channel * expansion)
        self.bn2 = nn.BatchNorm2d(in_channel * expansion)

        self.conv3 = nn.Conv2d(in_channel * expansion, out_channel, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU6(inplace=True)
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


class MobileNet_V2(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNet_V2, self).__init__()
        self.inplanes = 32
        self.model_name = 'MobileNet_V2'
        block = InvertedBottleneck
        layer_list = [1, 2, 3, 4, 3, 3, 1]

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layer_list[0], stride=1, expansion=1)
        self.layer2 = self._make_layer(block, 24, layer_list[1], stride=2, expansion=6)
        self.layer3 = self._make_layer(block, 32, layer_list[2], stride=2, expansion=6)
        self.layer4 = self._make_layer(block, 64, layer_list[3], stride=2, expansion=6)
        self.layer5 = self._make_layer(block, 96, layer_list[4], stride=1, expansion=6)
        self.layer6 = self._make_layer(block, 160, layer_list[5], stride=2, expansion=6)
        self.layer7 = self._make_layer(block, 320, layer_list[6], stride=1, expansion=6)

        self.conv8 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.conv9 = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, expansion):

        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

        layers = list()
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, expansion=expansion))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)

        x = self.conv8(x)
        x = self.avgpool(x)
        x = self.conv9(x)
        x = x.view(x.size(0), -1)
        return x

    def get_name(self):
        return self.model_name

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenet_v2(classes, pretrained=False, pretrained_path=None):
    model = MobileNet_V2(num_classes=classes)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    return model
