import math
import torch
import torch.nn as nn


class VGGNet(nn.Module):

    def __init__(self, layer_num, num_classes=1000):
        super(VGGNet, self).__init__()

        self.model_name = 'VGGNet_{}'.format(layer_num)
        self.layer_num = layer_num

        # Block1
        self.block1_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.block1_relu1 = nn.ReLU(True)
        self.block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.block1_relu2 = nn.ReLU(True)
        self.block1_mpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block2
        self.block2_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.block2_relu1 = nn.ReLU(True)
        self.block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.block2_relu2 = nn.ReLU(True)
        self.block2_mpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block3
        self.block3_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.block3_relu1 = nn.ReLU(True)
        self.block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block3_relu2 = nn.ReLU(True)
        self.block3_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block3_relu3 = nn.ReLU(True)
        self.bblock3_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.block3_relu4 = nn.ReLU(True)
        self.block3_mpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block4
        self.block4_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.block4_relu1 = nn.ReLU(True)
        self.block4_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block4_relu2 = nn.ReLU(True)
        self.block4_conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block4_relu3 = nn.ReLU(True)
        self.block4_conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block4_relu4 = nn.ReLU(True)
        self.block4_mpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block5
        self.block5_conv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_relu1 = nn.ReLU(True)
        self.block5_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_relu2 = nn.ReLU(True)
        self.block5_conv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_relu3 = nn.ReLU(True)
        self.block5_conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.block5_relu4 = nn.ReLU(True)
        self.block5_mpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

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

    def forward(self, x):
        x = self.block1_conv1(x)
        x = self.block1_relu1(x)
        if self.layer_num == 13 or self.layer_num == 16 or self.layer_num == 19:
            x = self.block1_conv2(x)
            x = self.block1_relu2(x)
        x = self.block1_mpool(x)

        # Block2
        x = self.block2_conv1(x)
        x = self.block2_relu1(x)
        if self.layer_num == 13 or self.layer_num == 16 or self.layer_num == 19:
            x = self.block2_conv2(x)
            x = self.block2_relu2(x)
        x = self.block2_mpool(x)

        # Block3
        x = self.block3_conv1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_relu2(x)
        if self.layer_num == 16 or self.layer_num == 19:
            x = self.block3_conv3(x)
            x = self.block3_relu3(x)
        if self.layer_num == 19:
            x = self.block3_conv4(x)
            x = self.block3_relu4(x)
        x = self.block3_mpool(x)

        # Block4
        x = self.block4_conv1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_relu2(x)
        if self.layer_num == 16 or self.layer_num == 19:
            x = self.block4_conv3(x)
            x = self.block4_relu3(x)
        if self.layer_num == 19:
            x = self.block4_conv4(x)
            x = self.block4_relu4(x)
        x = self.block4_mpool(x)

        # Block5
        x = self.block5_conv1(x)
        x = self.block5_relu1(x)
        x = self.block5_conv2(x)
        x = self.block5_relu2(x)
        if self.layer_num == 16 or self.layer_num == 19:
            x = self.block5_conv3(x)
            x = self.block5_relu3(x)
        if self.layer_num == 19:
            x = self.block5_conv4(x)
            x = self.block5_relu4(x)
        x = self.block5_mpool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
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


def vggnet(layer_num, classes, pretrained=False, pretrained_path=None):
    model = VGGNet(layer_num=layer_num, num_classes=classes)
    if pretrained:
        model.load_state_dict(torch.load(pretrained_path)['state_dict'])
    return model
