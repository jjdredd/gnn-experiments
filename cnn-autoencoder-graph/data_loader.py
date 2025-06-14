#! /usr/bin/env python3

import os
import re
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
from torchvision.io import decode_image

# Support only square images for now

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
                ((int(value[1]), int(value[2])), (int(value[3]), int(value[4]))))


    def __len__(self):
        return len(self.graph_edges)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, f'line_{idx}.png')
        image = decode_image(img_path)
        graph = torch.zeros(self.latent_size, self.latent_size)
        i = self.GridToIndex(self.graph_edges[idx][0])
        j = self.GridToIndex(self.graph_edges[idx][1])
        graph[i, j] = 1.0
        graph[j, i] = 1.0
        # graph[i, i] = 1.0
        # graph[j, j] = 1.0
        # we don't need vertex features, for now
        return { 'image': image[0].to(torch.float32).unsqueeze(0),
                 'graph' : graph }

    def GridToIndex(self, c: tuple[int, int]) -> int:
        return self.image_shape[0] * c[0] - c[1] + self.image_shape[1] - 1

    def GetImageShape(self):
        return self.image_shape

    def GetLatentSize(self):
        return self.latent_size

    @staticmethod
    def LatentToImageIndices(image_shape, latent_index):
        return (latent_index // int(image_shape[0]), latent_index % int(image_shape[0]))
