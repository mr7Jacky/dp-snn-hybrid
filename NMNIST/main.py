import torch
import snntorch as snn
from snntorch import surrogate
from snntorch import utils
import torch.nn as nn
import tonic
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import tonic.transforms as transforms
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
seed = 1024
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Basic training parameters
num_epochs = 80
batch_size = 128
lr = 1e-3
out_dim = 10

# LIF neuron parameters
beta = 0.5
spike_grad = surrogate.atan()
rst='subtract'

torch.autograd.set_detect_anomaly(True)
print(device)
# neuron and simulation parameters

# ===================
# ==== Data Prep ====
# ===================
sensor_size = tonic.datasets.NMNIST.sensor_size

frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=1000)])

trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
#trainset= torch.utils.data.Subset(trainset, np.arange(0, len(trainset), 10))
testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)
#testset = torch.utils.data.Subset(testset, np.arange(0, len(testset), 10))
# create dataloaders
trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
testloader = DataLoader(testset, shuffle=False, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

# from opacus.data_loader import DPDataLoader

# trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
# testloader = DataLoader(testset, shuffle=False, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())
# trainloader = DPDataLoader.from_data_loader(trainloader, distributed=False)

# ===================
# ==== Model ========
# ===================
from model import ConvNet
model = ConvNet().to(device) 

def forward_pass(model, data):
    spk_rec = torch.zeros((data.size(1), out_dim)).to(device)
    # utils.reset(model)  # resets hidden states for all LIF neurons in model
    mem1, mem2, mem3 = model.lif1.init_leaky(),  model.lif2.init_leaky(),  model.lif3.init_leaky()
    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out, mem1, mem2, mem3 = model(data[step], mem1, mem2, mem3)
        spk_rec += spk_out

    return spk_rec

# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
loss_fn =   torch.nn.CrossEntropyLoss()# SF.ce_count_loss()#SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

loss_hist = []
acc_hist = []

# Parameters for privacy engine
target_epsilon = 8
target_delta = 1e-2
max_grad_norm = 10 #1.0
r = 5

sample_rate = 1 / len(trainloader)
expected_batch_size = int(len(trainloader.dataset) * sample_rate)

from opacus.optimizers import DPOptimizer
from opacus.accountants.utils import get_noise_multiplier
from opacus.accountants import RDPAccountant
accountant = RDPAccountant()

noise_multiplier = get_noise_multiplier(
    target_epsilon=target_epsilon,
    target_delta=target_delta,
    sample_rate=sample_rate,
    epochs=num_epochs,
    accountant=accountant.mechanism()
)

optimizer = DPOptimizer(
    optimizer=optimizer,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm,
    expected_batch_size=expected_batch_size,
)

optimizer.attach_step_hook(
    accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
)

# training loop
for epoch in range(num_epochs):
    hidden_activations = []
    batch_labels = []
    
    # ===============
    # ==== Train ====
    # ===============
    model.train(True)
    
    correct, total = 0,0
    with tqdm(trainloader, unit="batch") as tepoch:
        for data, target in tepoch:
            data = data.to(device).permute(1, 0, 2, 3, 4)
            # print(data.shape)
            target = target.to(device)
            model.zero_grad()
            spk_rec = forward_pass(model, data)
            loss_val = loss_fn(spk_rec.float(), target)
            loss_val.backward()
            # for param in model.parameters():
            #     print(param.grad) # = torch.stack(param.accumulated_grads, dim=0)

            optimizer.step() 
            optimizer.zero_grad()
            spk_rec = torch.argmax(spk_rec, axis=1)
            correct += torch.eq(spk_rec, target).sum()
            total += target.size(0)
            # Store loss history for future plotting
            loss_hist.append(loss_val.item())
    print(f"Epoch {epoch} \nTrain Loss: {loss_val.item():.2f}")
    print(f"Privacy Bound: {accountant.get_epsilon(target_delta)}")
    print(f'Training Accuracy: {correct/total}')

    # ==============
    # ==== Test ====
    # ==============
    test_loss = 0.0
    correct, total = 0,0
    model.train(False)
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device).permute(1, 0, 2, 3, 4), target.to(device)
            spk_rec = forward_pass(model, data)
            loss = loss_fn(spk_rec.float(), target)
            spk_rec = torch.argmax(spk_rec, axis=1)
            test_loss += loss.item() * data.size(0)
            correct += torch.eq(spk_rec, target).sum()
            total += target.size(0)
    print(f'Testing Loss:{test_loss/len(testloader)}')
    print(f'Correct Predictions: {correct/total}')




# with BatchMemoryManager(
#         data_loader=data_loader,
#         max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
#         optimizer=optimizer
# ) as memory_safe_data_loader:
#     # if 1:
#     for epoch in range(1, epochs + 1):
#         result_train.append(train(model, device, memory_safe_data_loader, optimizer, epoch, privacy_engine))
#         result.append(test(model, device, test_loader))
#         scheduler.step()

# Goal:  97.78% accuracy
"""
For the CIFAR10 dataset, we train the DPSNN to ε = 8 in 80 epochs,
the gradient norm bound R = 6, the batch size B = 1024, and the learning
rate is set to 0.001. For MNIST and Fashion-MNIST dataset, we train the
DPSNN to ε = 3 in 20 epochs, the gradient norm bound R = 2, the batch
size B = 1024, and the learning rate is set to 0.005. The DPSNN can achieve
97.71% mean test accuracy on MNIST, 85.72% on Fashion-MNIST, and
65.70% on CIFAR10.
"""
