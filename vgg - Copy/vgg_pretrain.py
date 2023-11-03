

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from opacus import GradSampleModule
from opacus.validators import ModuleValidator
from torchvision.models import vgg16, VGG16_Weights
import copy

class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        
        beta = 0.5
        thr = 0.5
        alpha = 0.9
        
        pretrain = vgg16(weights=VGG16_Weights.DEFAULT)
        self.features = []
        for i, p in enumerate(pretrain.features):
            self.features.append(copy.deepcopy(p))
            if isinstance(p, nn.MaxPool2d):
                self.features.append(snn.Synaptic(alpha=alpha, beta=beta, threshold=thr, 
                                        reset_mechanism='subtract', init_hidden=True))
        self.features = GradSampleModule(nn.Sequential(*self.features))
        
        self.classifier = []
        for i, p in enumerate(pretrain.classifier):
            self.classifier.append(copy.deepcopy(p))
            if isinstance(p, nn.Dropout):
                self.classifier.append(snn.Synaptic(alpha=alpha, beta=beta, threshold=thr, 
                                        reset_mechanism='subtract', init_hidden=True))
        self.classifier[-1] = nn.Linear(self.classifier[-1].weight.data.shape[-1], num_classes)
        self.classifier = GradSampleModule(nn.Sequential(*self.classifier))
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
       
        self.lifo = snn.Synaptic(alpha=0.9, beta=0.5, threshold=1, 
                                reset_mechanism='subtract', init_hidden=True, output=True)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.lifo(x)
        return out

a = VGG()