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
            nn.Conv2d(in_channels=3, out_channels=48,
                      kernel_size=(5, 5),
                      stride=1, padding=0, bias=True),
            nn.ELU(),
            nn.Conv2d(in_channels=48, out_channels=144,
                      kernel_size=(5, 5),
                      stride=1, padding=0, bias=True),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # input 12x12, output ((12 - 6)# / 2)
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=144, out_channels=288,
                      kernel_size=(3, 3),
                      stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        # input 3x3x240 (out_channels), flattening
        self.fully_connected_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=7200, out_features=3000),
            nn.ReLU(),
            nn.Linear(in_features=3000, out_features=1000),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=classes),
	    nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        logits = self.fully_connected_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
