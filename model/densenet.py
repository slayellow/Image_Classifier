import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# add_module : 이 함수를 통해 Layer의 이름을 설정할 수 있다.

class DenseLayer(nn.Sequential):

    def __init__(self, in_channel, growth_rate):
        super(DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channel))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channel, 4 * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(4 * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Sequential):

    def __init__(self, repeat, in_channel, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(repeat):
            layer = DenseLayer(in_channel + i * growth_rate, growth_rate)
            self.add_module('denselayer_%d' % (i + 1), layer)


class Transition(nn.Sequential):

    def __init__(self, in_channel, out_channel):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channel))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(self, layer_num, num_classes=1000, growth_rate=32):
        super(DenseNet, self).__init__()
        self.model_name = 'DenseNet_{}'.format(layer_num)
        self.growth_rate = growth_rate
        self.inplanes = 2 * growth_rate
        blocks = {121: (6, 12, 24, 16),
                  169: (6, 12, 32, 32),
                  201: (6, 12, 48, 32),
                  264: (6, 12, 64, 48)}

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(self.inplanes)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        for i, repeat in enumerate(blocks[layer_num]):
            block = DenseBlock(repeat, self.inplanes, growth_rate=self.growth_rate)
            self.features.add_module('denseblock_%d' % (i + 1), block)
            self.inplanes = self.inplanes + repeat * growth_rate
            if i != len(blocks[layer_num]) - 1:
                trans = Transition(in_channel=self.inplanes, out_channel=self.inplanes // 2)
                self.features.add_module('transition_%d' % (i + 1), trans)
                self.inplanes = self.inplanes // 2

        self.features.add_module('norm5', nn.BatchNorm2d(self.inplanes))
        self.classifier = nn.Linear(self.inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def get_name(self):
        return self.model_name

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


def densenet(layer_num, classes, pretrained=False, pretrained_path=None):
    model = DenseNet(layer_num=layer_num, num_classes=classes)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    return model
