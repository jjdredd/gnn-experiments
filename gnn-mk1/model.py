import torch
import torch.nn as nn
import torch_geometric.nn as tgnn

class CnnGraphEncoder_Ntc_Wes(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        # input 16x16,
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), # AvgPool2d
            #  output 8x8 (16/2)
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.Conv2d(in_channels=16, out_channels=8,
                      kernel_size=(3, 3),
                      stride=1, padding=1, bias=True),
            nn.Softsign(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        # 4x4
        self.gc_1 = tgnn.GCNConv(8, 4)

    def forward(self, x):
        x = self.conv(x)
        edges = 
        vertices = 
        vertices = self.gcn_1(vertices)
        return 

