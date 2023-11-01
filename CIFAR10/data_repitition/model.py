import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
import torch
import torch.nn as nn
import torch.nn.functional as F

from opacus import GradSampleModule
from einops import rearrange
# Define Network

# LIF neuron parameters
beta = 0.5
threshold = 1.0
spike_grad = surrogate.atan()
rst='subtract'

class TEP(nn.Module):
    def __init__(self, step, channel, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TEP, self).__init__()
        self.step = step
        self.gn = nn.GroupNorm(channel, channel)


    def forward(self, x):

        x = rearrange(x, '(t b) c w h -> t b c w h', t=self.step)
        fire_rate = torch.mean(x, dim=0)
        fire_rate = self.gn(fire_rate) + 1

        x = x * fire_rate
        x = rearrange(x, 't b c w h -> (t b) c w h')

        return x

class ConvNet(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        # Initialize layers
        # self.conv1 = GradSampleModule(nn.Conv2d(in_channels, 32, kernel_size=5, padding=2))
        # self.lif1 = snn.Leaky(beta=beta, reset_mechanism=rst, spike_grad=spike_grad)
        # self.conv2 = GradSampleModule(nn.Conv2d(32, 64, kernel_size=5, padding=0))
        # self.lif2 = snn.Leaky(beta=beta, reset_mechanism=rst, spike_grad=spike_grad)
        # self.fc1 = GradSampleModule(nn.Linear(576, 120))
        # self.lif3 = snn.Leaky(beta=beta, reset_mechanism=rst, spike_grad=spike_grad)
        # self.fc2 = GradSampleModule(nn.Linear(120, 84))
        # self.lif4 = snn.Leaky(beta=beta, reset_mechanism=rst, spike_grad=spike_grad)
        # self.fc3 = GradSampleModule(nn.Linear(84, 10))
        # self.lif5 = snn.Leaky(beta=beta, reset_mechanism=rst, spike_grad=spike_grad, output=True)

        self.kernel_size = 3

        self.classifier = GradSampleModule(nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=5, padding=1),
                                        nn.ReLU(),
                                        torch.nn.AvgPool2d(2,2),
                                        nn.LayerNorm([15, 15]),
                                        snn.Synaptic(alpha=0.9, beta=beta, threshold=threshold, 
                                        reset_mechanism=rst, spike_grad=spike_grad, init_hidden=True),
                                        
                                        nn.Conv2d(64, 128, kernel_size=5, padding=1),
                                        nn.ReLU(),
                                        torch.nn.AvgPool2d(2,2),
                                        nn.LayerNorm([6, 6]),
                                        snn.Synaptic(alpha=0.9, beta=beta, threshold=threshold, 
                                        reset_mechanism=rst, spike_grad=spike_grad, init_hidden=True),
                                        ))

        self.fc3 = GradSampleModule(nn.Linear(4608, 10))
        self.lif5 = snn.Synaptic(alpha=0.9, beta=beta, threshold=threshold,
        reset_mechanism=rst, spike_grad=spike_grad, init_hidden=True, output=True)
        self.init_weights()

    def init_weights(self):
        for i, w in enumerate(self.parameters()):
            if len(w.shape) < 2:
                torch.nn.init.normal_(w)
            else:
                torch.nn.init.xavier_normal_(w)
        

    def forward(self, x):
        batch_size = x.shape[0]
        # Initialize hidden states and outputs at t=0
        cur1 = self.classifier(x)
        cur5 = self.fc3(cur1.reshape(batch_size, -1))
        spk5, mem5, _  = self.lif5(cur5)
        return spk5, mem5
