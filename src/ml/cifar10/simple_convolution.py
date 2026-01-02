import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, input):
         # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))

        # Subsampling layer S2: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a ( N, 6, 14, 14) Tensor
        s2 = F.max_pool2d(c1, (2, 2))
        
        # Convulation layer C3: 6 input channels, 6 output channels, 
        # 5x5 sqaure convolution, it uses RELU activation function, and 
        # outputs a (N, 16, 10, 10) Tensor
        c3 = F.relu(self.conv2(s2))

        # Subsampling layer 4: 2x2 grid, purely functional,
        # this layer does not have any parameter, and outputs a ( N, 16, 5, 5) Tensor
        s4 = F.max_pool2d(c3, 2)

        # Flatten operation: purely functional, outputs a (n, 16* 5 * 5) Tensor
        s4 = torch.flatten(s4, 1) # flatten all dimensions except batch

        # Fully connected layer F5: (N, 400) Tensor input,
        # and outputs a (N, 120) Tensor, it uses RELU activation function
        f5 = F.relu(self.fc1(s4))

        # Fully connected layer F6: (N, 120) Tensor input,
        # and outputs a (N, 84) Tensor, it uses RELU activation function
        f6 = F.relu(self.fc2(f5))

        # Fully connected layer Output: (N, 84) Tensor input, and outputs a (N, 10) Tensor
        output = self.fc3(f6)
        return output


# create your optimizer
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()
print(net)

params = list(net.parameters())
print(f'Total number of parameters: {len(params)}')
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
target = torch.randn(10)
target = target.view(1, -1)  # make it size (1, 10)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
loss.backward()
optimizer.step()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)



