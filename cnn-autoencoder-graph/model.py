import torch
from torch import nn
from torchvision.transforms import ToTensor

class CnnGraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 32x32, output 14x14 ((32 - 8)/2)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=8,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=4,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layer_1(x) #x.unsqueeze(0))
        x = torch.flatten(x, start_dim=2, end_dim=3)
        return self.sigmoid(torch.einsum('...ij,...ik->...jk', x, x))


class CnnGraphEncoderDeconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 32x32,
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=0, bias=True),
            # nn.Softsign(),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # AvgPool2d
            #  output 14x14 ((32 - 2)/2)
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=(3, 3),
                      stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        # 6x6    (14 - 2) /2
        self.transposed_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=(5, 5),
                               stride=2, padding=1, bias=True),
            nn.Softsign(),
            # (6 - 1) * 2 + (5 - 1) + 1 - 2 = 13
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=(7, 7),
                               stride=2, padding=1, bias=True),
            nn.Softsign(),
            # (13 - 1) * 2 + (7 - 1) + 1 - 2 = 29
            nn.ConvTranspose2d(in_channels=16, out_channels=8,
                               kernel_size=(4, 4),
                               stride=1, bias=True),
            nn.Softsign(),
            # (29 - 1) + 4 - 1 + 1 = 32
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.transposed_conv(x)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        return self.sigmoid(torch.einsum('...ij,...ik->...jk', x, x))




class CnnGraphEncoderDeconvLong(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 16x16,
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            # nn.Softsign(),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # AvgPool2d
            #  output 8x8 (16/2)
            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        # 4x4
        self.transposed_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=32,
                               kernel_size=(7, 7),
                               stride=1, padding=1, bias=True),
            nn.Softsign(),
            # (4 - 1) * 1 + (7 - 1) + 1 - 2 = 8
            nn.ConvTranspose2d(in_channels=32, out_channels=64,
                               kernel_size=(5, 5),
                               stride=1, padding=1, bias=True),
            nn.Softsign(),
            # (8 - 1) * 1 + (5 - 1) + 1 - 2 = 10
            nn.ConvTranspose2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5),
                               stride=1, bias=True),
            nn.Softsign(),
            # (10 - 1) + 5 - 1 + 1 = 14
            nn.ConvTranspose2d(in_channels=128, out_channels=256,
                               kernel_size=(3, 3),
                               stride=1, bias=True),
            nn.Softsign(),
            # (14 - 1) + 3 - 1 + 1 = 16
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.transposed_conv(x)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        return self.sigmoid(torch.einsum('...ij,...ik->...jk', x, x))



class CnnGraphEncoderNoEinsum(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 16x16,
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            # nn.Softsign(),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # AvgPool2d
            #  output 8x8 (16/2)
            nn.Conv2d(in_channels=8, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        # 4x4
        self.transposed_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=32,
                               kernel_size=(7, 7),
                               stride=1, padding=1, bias=True),
            nn.ReLU(),
            # (4 - 1) * 1 + (7 - 1) + 1 - 2 = 8
            nn.ConvTranspose2d(in_channels=32, out_channels=64,
                               kernel_size=(5, 5),
                               stride=1, padding=1, bias=True),
            nn.ReLU(),
            # (8 - 1) * 1 + (5 - 1) + 1 - 2 = 10
            nn.ConvTranspose2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5),
                               stride=1, bias=True),
            nn.ReLU(),
            # (10 - 1) + 5 - 1 + 1 = 14
            nn.ConvTranspose2d(in_channels=128, out_channels=256,
                               kernel_size=(3, 3),
                               stride=1, bias=True),
            nn.Sigmoid(),
            # (14 - 1) + 3 - 1 + 1 = 16
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.transposed_conv(x)
        return torch.flatten(x, start_dim=2, end_dim=3)



class CnnGraphEncoderNoEinsumLong(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 16x16,
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            # nn.Softsign(),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # AvgPool2d
            #  output 8x8 (16/2)
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        # 4x4
        self.transposed_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128,
                               kernel_size=(5, 5),
                               stride=1, padding=0, bias=True),
            nn.ReLU(),
            # (4 - 1) * 1 + (5 - 1) + 1 - 0 = 8
            nn.ConvTranspose2d(in_channels=128, out_channels=128,
                               kernel_size=(5, 5),
                               stride=1, padding=0, bias=True),
            nn.ReLU(),
            # (8 - 1) * 1 + (5 - 1) + 1 - 0 = 12
            nn.ConvTranspose2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5),
                               stride=1, bias=True),
            # (12 - 1) + 5 - 1 + 1 = 16
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.transposed_conv(x)
        return torch.flatten(x, start_dim=2, end_dim=3)


# no eigensum, no transposed convolution
class CnnGraphEncoderNesNtc(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 16x16,
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            # nn.Softsign(),
            nn.ReLU(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # AvgPool2d
            #  output 8x8 (16/2)
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        # 4x4
        self.transposed_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 8x8
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(5, 5),
                               stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 16x16
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.transposed_conv(x)
        return torch.flatten(x, start_dim=2, end_dim=3)



class CnnGraphEncoderNesNtcNp(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 32x32, output 14x14 ((32 - 8)/2)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv_layer_1(x)
        return torch.flatten(x, start_dim=2, end_dim=3)


# no transposed convolution, with einsum
class CnnGraphEncoder_Ntc_Wes(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 16x16,
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            # nn.Softsign(),
            nn.Softsign(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # AvgPool2d
            #  output 8x8 (16/2)
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        # 4x4
        self.transposed_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 8x8
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=16, out_channels=16,
                               kernel_size=(5, 5),
                               stride=1, padding=2, bias=True),
            nn.Softsign(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 16x16
            nn.Conv2d(in_channels=16, out_channels=16,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=16, out_channels=8,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.transposed_conv(x)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        return self.sigmoid(-torch.einsum('...ij,...ik->...jk', x, x))


# no transposed convolution, with einsum 64x64 input
class CnnGraphEncoder_Ntc_Wes_64(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 64x64,
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            # nn.Softsign(),
            nn.Softsign(),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # AvgPool2d
            #  output 32x32 (64/2)
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        # 16x16
        self.transposed_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 32x32
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(5, 5),
                               stride=1, padding=2, bias=True),
            nn.Softsign(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            # 64x64
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(5, 5),
                      stride=1, padding=2, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.transposed_conv(x)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        return self.sigmoid(-torch.einsum('...ij,...ik->...jk', x, x))

