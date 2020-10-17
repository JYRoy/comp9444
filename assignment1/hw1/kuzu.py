# kuzu.py
# COMP9444, CSE, UNSW

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.linear = nn.Linear(in_features=28*28, out_features=10, bias=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return F.log_softmax(x)

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        self.linear1 = nn.Linear(in_features=28*28, out_features=300, bias=True)
        self.linear2 = nn.Linear(in_features=300, out_features=100, bias=True)
        self.linear3 = nn.Linear(in_features=100, out_features=10, bias=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.tanh(self.linear1(x))
        x = F.tanh(self.linear2(x))
        x = F.log_softmax(self.linear3(x))
        return x

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.maxpool2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=50 * 5 * 5, out_features=512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.maxpool1(x)
        x = self.conv2d2(x)
        x = self.maxpool2(x)
        x = x.view(x.shape[0], -1)
        x = self.linear1(x)
        return F.log_softmax(x)
