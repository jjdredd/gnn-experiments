import torch
from torch import nn
from torchvision.transforms import ToTensor
from time import perf_counter_ns

import data_loader as dl


class CnnGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # input 32x32, output 14x14 ((32 - 8)/2)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=(5, 5),
                      stride=1, padding=0, bias=True),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=(5, 5),
                      stride=1, padding=0, bias=True),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3),
                      stride=1, padding=0, bias=True),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=8,
                      kernel_size=(3, 3),
                      stride=1, padding=0, bias=True),
            nn.ELU(),
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        
        return x

model = NeuralNetwork().to(device)
print(model)
