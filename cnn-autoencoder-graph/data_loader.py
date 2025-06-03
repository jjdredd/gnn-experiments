#! /usr/bin/env python3

import os
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
from torchvision.io import read_image

# 

class LineGraphDataset(Dataset):
    def __init__(self, data_directory : str):
        self.directory : str = data_directory
        self.graph_vertices = []
        self.graph_edges = []
        self.image
        line_file = open(os.path.join(self.directory, f'{self.directory}/graphs.txt'))
        lines = [line.rstrip() for line in line_file]
        for i in range(len(lines)):
            value = re.split(r'\t', lines[i]))
            self.graph_edges.append(torch.tensor([[value[0], value[1]], [value[3], value[4]]]))
    
    def __len__(self):
        return len(self.graph_edges)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, f'line_{i}.png')
        image = read_image(img_path)
        return { 'image': image,
                 'vertex_features': self.graph_vertices[idx],
                 'positive_edges' : self.graph_edges[idx],
                 'negative_edges' : self.graph_edges[idx] }

    @staticmethod
    def GridToIndex(i: int, j :int) -> int:
        return


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
