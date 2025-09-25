import torch
import torch.nn as nn
import torch_geometric.nn as tgnn

class CnnGraphEncoder_Ntc_Wes(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        # input 16x16,
        self.layers = [tgnn.GCNConv(in_channels=, out_channels=),
                       tgnn.TopKPooling(in_channels=, ration=0.5),
                       tgnn.GCNConv(in_channels=, out_channels=),
                       tgnn.TopKPooling(in_channels=, ration=0.5),
                       tgnn.GCNConv(in_channels=, out_channels=)]

    def forward(self, x):
        edges = 
        vertices =
        # this is wrong, pooling and conv layers have different i/o
        for l in self.layers:
            edges, vertices, _ = l(edges, vertices)
        return edges, vertices

