import torch
from torch import nn
from torchvision.transforms import ToTensor

class CnnGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 32x32, output 14x14 ((32 - 8)/2)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=16, out_channels=8,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=8, out_channels=2,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layer_1(x) #x.unsqueeze(0))
        x = torch.flatten(x, start_dim=1, end_dim=2)
        return self.sigmoid(torch.einsum('...ij,...ik->...jk', x, x))

