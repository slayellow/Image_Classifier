import math
import torch
import torch.nn as nn


class MobileNet_V1(nn.Module):

    def __init__(self, num_classes=1000):
        super(MobileNet_V1, self).__init__()

        self.model_name = 'MobileNet_V1'

        self.layer1 = self.make_conv_layer(in_channel=3, out_channel=32, kernel=3, stride=2)
        self.layer2 = self.make_dw_conv_layer(in_channel=32, out_channel=64, stride=1)
        self.layer3 = self.make_dw_conv_layer(in_channel=64, out_channel=128, stride=2)
        self.layer4 = self.make_dw_conv_layer(in_channel=128, out_channel=128, stride=1)
        self.layer5 = self.make_dw_conv_layer(in_channel=128, out_channel=256, stride=2)
        self.layer6 = self.make_dw_conv_layer(in_channel=256, out_channel=256, stride=1)
        self.layer7 = self.make_dw_conv_layer(in_channel=256, out_channel=512, stride=2)

        self.layer8 = self.make_dw_conv_layer(in_channel=512, out_channel=512, stride=1)
        self.layer9 = self.make_dw_conv_layer(in_channel=512, out_channel=512, stride=1)
        self.layer10 = self.make_dw_conv_layer(in_channel=512, out_channel=512, stride=1)
        self.layer11 = self.make_dw_conv_layer(in_channel=512, out_channel=512, stride=1)
        self.layer12 = self.make_dw_conv_layer(in_channel=512, out_channel=512, stride=1)

        self.layer13 = self.make_dw_conv_layer(in_channel=512, out_channel=1024, stride=2)
        self.layer14 = self.make_dw_conv_layer(in_channel=1024, out_channel=1024, stride=1)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(1024, num_classes)

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

    def make_conv_layer(self, in_channel, out_channel, kernel, stride):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        return layer

    def make_dw_conv_layer(self, in_channel, out_channel, stride):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        return layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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


def mobilenet_v1(classes, pretrained=False, pretrained_path=None):
    model = MobileNet_V1(num_classes=classes)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    return model
