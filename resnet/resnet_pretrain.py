

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.planes = planes
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


from opacus import GradSampleModule
from opacus.validators import ModuleValidator
from torchvision.models import resnet18, ResNet18_Weights

class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(ResNet, self).__init__()

        pretrain = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = pretrain.conv1.weight.data.clone()
        self.conv1 = GradSampleModule(self.conv1)
        self.bn1 =  GradSampleModule(nn.GroupNorm(32,64))

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer1.load_state_dict(pretrain.layer1.state_dict())
        self.layer1 = GradSampleModule(ModuleValidator.fix(self.layer1))

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer2.load_state_dict(pretrain.layer2.state_dict())
        self.layer2 = GradSampleModule(ModuleValidator.fix(self.layer2))

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer3.load_state_dict(pretrain.layer3.state_dict())
        self.layer3 = GradSampleModule(ModuleValidator.fix(self.layer3))

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer4.load_state_dict(pretrain.layer4.state_dict())
        self.layer4 = GradSampleModule(ModuleValidator.fix(self.layer4))

        self.linear = GradSampleModule(nn.Linear(512, num_classes))
        # self.linear.weight.data = pretrain.fc.weight.data.clone()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.lif = [snn.Synaptic(alpha=0.9, beta=0.5, threshold=1, 
                                reset_mechanism='subtract', init_hidden=True) for _ in range(5)]
        self.lifo = snn.Synaptic(alpha=0.9, beta=0.5, threshold=1, 
                                reset_mechanism='subtract', init_hidden=True, output=True)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.lif[0](out)
        out = self.layer1(out)
        out = self.lif[1](out)
        out = self.layer2(out)
        out = self.lif[2](out)
        out = self.layer3(out)
        out = self.lif[3](out)
        out = self.layer4(out)
        out = self.lif[4](out)
        out = F.avg_pool2d(out, 2)
        out = self.flatten(out)
        out = self.linear(out)
        out, _ ,_ = self.lifo(out)
        return out