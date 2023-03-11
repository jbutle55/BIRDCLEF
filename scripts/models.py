import torch
from torch import nn


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 35, kernel_size=80, stride=16)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        return x
