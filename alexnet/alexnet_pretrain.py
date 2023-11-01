

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn


from opacus import GradSampleModule
from opacus.validators import ModuleValidator
from torchvision.models import alexnet, AlexNet_Weights

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        
        pretrain = alexnet(weights=AlexNet_Weights.DEFAULT)
        # SNN Param
        beta = 0.5
        thr = 0.5
        alpha = 0.9
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11,11), stride=(4,4), padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
            snn.Synaptic(alpha=alpha, beta=beta, threshold=thr, 
                        reset_mechanism='subtract', init_hidden=True),
            
            nn.Conv2d(64, 192, kernel_size=(5,5), stride=(1,1), padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
            snn.Synaptic(alpha=alpha, beta=beta, threshold=thr, 
                        reset_mechanism='subtract', init_hidden=True),
            
            nn.Conv2d(192, 384, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            snn.Synaptic(alpha=alpha, beta=beta, threshold=thr, 
                        reset_mechanism='subtract', init_hidden=True),
            
            nn.Conv2d(384, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            snn.Synaptic(alpha=alpha, beta=beta, threshold=thr, 
                        reset_mechanism='subtract', init_hidden=True),
            
            nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0),
            snn.Synaptic(alpha=alpha, beta=beta, threshold=thr, 
                        reset_mechanism='subtract', init_hidden=True),
        )
        pretrain_feature_idx = [0,3,6,8,10]
        feature_idx = [0,4,8,11,14]
        for i,j in zip(pretrain_feature_idx, feature_idx):
            self.features[j].weight.data = pretrain.features[i].weight.data.clone()
        
        self.features = GradSampleModule(self.features)
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            snn.Synaptic(alpha=alpha, beta=beta, threshold=thr, 
                        reset_mechanism='subtract', init_hidden=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            snn.Synaptic(alpha=alpha, beta=beta, threshold=thr, 
                        reset_mechanism='subtract', init_hidden=True),
            nn.Linear(4096, num_classes),
        )
        pretrain_classifier_idx = [1,4]
        classifier_idx = [1,5]
        for i,j in zip(pretrain_classifier_idx, classifier_idx):
            self.classifier[j].weight.data = pretrain.classifier[i].weight.data.clone()
        
        
        self.classifier = GradSampleModule(self.classifier)
        self.lifo = snn.Synaptic(alpha=alpha, beta=beta, threshold=thr, 
                        reset_mechanism='subtract', init_hidden=True)
        del pretrain
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.lifo(x)
        return x