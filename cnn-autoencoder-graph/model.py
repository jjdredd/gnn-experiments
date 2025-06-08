import torch
from torch import nn
from torchvision.transforms import ToTensor
from time import perf_counter_ns

import data_loader as dl

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
else:
    device = torch.device("cpu")

print(f"Using {device} device")

class CnnGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
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
        x = torch.flatten(x, end_dim=1)
        return self.sigmoid(torch.einsum('i,j->ij', x, x))

model = NeuralNetwork().to(device)
print(model)
