#!/usr/bin/env python3

import torch
from torch import nn
from torchvision.transforms import ToTensor
from time import perf_counter_ns

import model
import data_loader as dl

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
else:
    device = torch.device("cpu")

print(f"Using {device} device")


GraphModel = model.CnnGraphEncoder().to(device)
print(model)
GraphModel.eval()


dataset = dl.LineGraphDataset('./train-ds')
with torch.no_grad():
    image = dataset[0]['image']
    gt_graph_tensor = dataset[0]['graph']
    print('Input image shape ', image.shape)
    print('GT tensor shape ', gt_graph_tensor.shape)
    output_graph = GraphModel(image.to(device))
    print('output graph tensor shape ', output_graph.shape)
