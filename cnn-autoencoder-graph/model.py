import torch
from torch import nn
from torchvision.transforms import ToTensor

class CnnGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 32x32, output 14x14 ((32 - 8)/2)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv_layer_1(x) #x.unsqueeze(0))
        x = torch.flatten(x, start_dim=2, end_dim=3)
        return self.sigmoid(torch.einsum('...ij,...ik->...jk', x, x))


class CnnGraphEncoderDeconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 32x32, output 14x14 ((32 - 4)/2)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(5, 5),
                      stride=1, padding=0, bias=True),
            nn.Softsign(),
            nn.AvgPool2d(kernel_size=(2, 2), strid=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(5, 5),
                      stride=1, padding=0, bias=True),
            nn.Softsign(),
            nn.AvgPool2d(kernel_size=(2, 2), strid=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layer_1(x) #x.unsqueeze(0))
        x = torch.flatten(x, start_dim=2, end_dim=3)
        return self.sigmoid(torch.einsum('...ij,...ik->...jk', x, x))

