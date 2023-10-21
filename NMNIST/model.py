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

# Define Network

# LIF neuron parameters
beta = 0.5
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
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.conv1 = GradSampleModule(nn.Conv2d(2, 32, kernel_size=7, padding=0))
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = GradSampleModule(nn.Conv2d(32, 64, kernel_size=4, padding=0))
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc1 = GradSampleModule(nn.Linear(100*4*4, 10))
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.init_weights()

    def init_weights(self):
        for i, w in enumerate(self.parameters()):
            if len(w.shape) < 2:
                torch.nn.init.uniform_(w)
            else:
                torch.nn.init.xavier_uniform_(w)
        
            if w.__class__.name == 'Linear':
                w.forbid_grad_accumulation()
            if w.__class__.name == 'Conv2d':
                w.forbid_grad_accumulation()

    def forward(self, x, mem1, mem2, mem3):
        batch_size = x.shape[0]
        # Initialize hidden states and outputs at t=0
        cur1 = F.avg_pool2d(self.conv1(x), 2, stride=2)
        spk1, mem1 = self.lif1(cur1, mem1)
            #TEP(step=step, channel=32),
        cur2 = F.avg_pool2d(self.conv2(spk1), 2, stride=2) #.avg_pool2d
        spk2, mem2 = self.lif2(cur2, mem2)
        cur3 = self.fc1(spk2.view(batch_size, -1))
        spk3, mem3 = self.lif3(cur3, mem3)

        return spk3, mem1, mem2, mem3
