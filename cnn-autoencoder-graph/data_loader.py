#! /usr/bin/env python3

import os
import re
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
from torchvision.io import decode_image

# 

class LineGraphDataset(Dataset):
    def __init__(self, data_directory : str):
        self.directory : str = data_directory
        self.graph_edges = []

        image = decode_image(os.path.join(self.directory, 'line_0.png'))
        self.image_shape = image[0].shape
        self.latent_size = image[0].shape[0] * image[0].shape[1]

        line_file = open(os.path.join(self.directory, 'graphs.txt'))
        lines = [line.rstrip() for line in line_file]
        for i in range(len(lines)):
            value = re.split(r'\t', lines[i])
            self.graph_edges.append(
                ((int(value[0]), int(value[1])), (int(value[2]), int(value[3]))))


    def __len__(self):
        return len(self.graph_edges)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, f'line_{idx}.png')
        image = decode_image(img_path)
        graph = torch.zeros(self.latent_size ** 2)
        edge_index_1 = self.GridToOutputIndex(self.graph_edges[idx][0],
                                              self.graph_edges[idx][1])
        graph[edge_index_1] = 1.0
        edge_index_2 = self.GridToOutputIndex(self.graph_edges[idx][1],
                                                          self.graph_edges[idx][0])
        graph[edge_index_2] = 1.0
        # we don't need vertex features, for now
        return { 'image': image[0].to(torch.float32),
                 'graph' : graph }

    def GridToIndex(self, i: int, j: int) -> int:
        return self.image_shape[0] * i + j

    def GridToOutputIndex(self, start: tuple[int, int], end: tuple[int, int]) -> int:
        start_index = self.GridToIndex(start[0], start[1])
        end_index = self.GridToIndex(end[0], end[1])
        return start_index * self.latent_size + end_index
