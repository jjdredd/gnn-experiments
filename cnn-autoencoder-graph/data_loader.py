#! /usr/bin/env python3

import os
import torch
from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
from torchvision.io import read_image

class LineGraphDataset(Dataset):
    def __init__(self, data_directory : str):
        self.directory : str = data_directory
        self.graph_vertices = []
        self.graph_edges = []
        line_file = open(os.path.join(self.directory, f'{self.directory}/graphs.txt'))
        lines = [line.rstrip() for line in line_file]
        for i in range(len(lines)):
            value = re.split(r'\t', lines[i]))
            self.graph_vertices.append(torch.tensor([[value[0], value[1]], [value[3], value[4]]]))
            self.graph_edges.append(torch.tensor([[0, 1], [1, 0]]))
    
    def __len__(self):
        return len(self.graph_edges)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, f'line_{i}.png')
        image = read_image(img_path)
        image_graph = self.ImageToGraph(image)
        return { 'image_vertices': image_graph[0],
                 'image_edges': image_graph[1],
                 'vertices': self.graph_vertices[idx],
                 'edges' : self.graph_edges[idx] }

    def ImageToGraph(self, image):
        pass
