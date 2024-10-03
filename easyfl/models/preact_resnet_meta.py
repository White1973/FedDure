'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from easyfl.models.meta_layers import *

class PreActBlockMeta(MetaModule):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlockMeta, self).__init__()
        self.bn1 = MetaBatchNorm2d(in_planes)
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MetaConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneckMeta(MetaModule):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneckMeta, self).__init__()
        self.bn1 = MetaBatchNorm2d(in_planes)
        self.conv1 = MetaConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                MetaConv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNetMeta(MetaModule):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNetMeta, self).__init__()
        self.in_planes = 64

        self.conv1 = MetaConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_max_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return F.log_softmax(self.linear(out), dim=1)
class WideResNet(MetaModule):
    def __init__(self, depth=28, widen_factor=2, n_classes=10, dropRate=0.0, transform_fn=None):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = MetaBasicBlock
        # 1st conv before any network block
        self.conv1 = MetaConv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = MetaNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = MetaNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = MetaNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = MetaBatchNorm2d(nChannels[3])
        self.relu = nn.LeakyReLU(0.1)
        self.fc = MetaLinear(nChannels[3], n_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, MetaBatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MetaLinear):
                #m.bias.data.zero_()
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self.transform_fn = transform_fn
    def forward(self, x):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, MetaBatchNorm2d):
                m.update_batch_stats = flag

def preact_resnet_meta18(): return PreActResNetMeta(PreActBlockMeta, [2,2,2,2])
def preact_resnet_meta2332(): return PreActResNetMeta(PreActBlockMeta, [2,3,3,2])
def preact_resnet_meta3333(): return PreActResNetMeta(PreActBlockMeta, [3,3,3,3])
def preact_resnet_meta34(): return PreActResNetMeta(PreActBlockMeta, [3,4,6,3])
def preact_resnet_meta50(): return PreActResNetMeta(PreActBottleneckMeta, [3,4,6,3])
def preActResNetMeta101(): return PreActResNetMeta(PreActBottleneckMeta, [3,4,23,3])
def preActResNetMeta152(): return PreActResNetMeta(PreActBottleneckMeta, [3,8,36,3])
class CNN(MetaModule):
    def __init__(self, n_out):
        super(CNN, self).__init__()

        self.conv = torch.nn.Sequential(MetaConv2d(3, 16, 1, padding=0),
                                        nn.ReLU()
                                        )
        self.dense = torch.nn.Sequential(nn.Dropout(p=0.5),
                                        MetaLinear(16 * 32 * 32, n_out))
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 32 * 16)
        x = self.dense(x)
        return x

#def Model(num_classes=10): return WideResNet()


def Meta_conv_bn_relu_pool(in_channel,out_channel,pool=False):
    layer=[
        MetaConv2d(in_channel,out_channel,3,padding=1),
        MetaBatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    ]
    if pool:
        layer.append(nn.MaxPool2d(2))
    return nn.Sequential(*layer)

class Model(MetaModule):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.prep = Meta_conv_bn_relu_pool(in_channels, 64)
        self.layer1_head = Meta_conv_bn_relu_pool(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(Meta_conv_bn_relu_pool(128, 128), Meta_conv_bn_relu_pool(128, 128))
        self.layer2 = Meta_conv_bn_relu_pool(128, 256, pool=True)
        self.layer3_head = Meta_conv_bn_relu_pool(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(Meta_conv_bn_relu_pool(512, 512), Meta_conv_bn_relu_pool(512, 512))
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            MetaLinear(512, num_classes))

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        x = self.classifier(x)

        return x
