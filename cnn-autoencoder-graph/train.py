#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.io import decode_image
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
print(GraphModel)

StdCrossEntropyLoss = nn.CrossEntropyLoss()
def AdjacencyCrossEntropy(prediction, ground_truth):
    return StdCrossEntropyLoss(prediction.flatten(), ground_truth.flatten())

# Support only square images for now
def Train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    print('Dataset size: ', size)
    model.train()
    for i, data in enumerate(dataloader):
        image = data['image'].to(device)
        graph = data['graph'].to(device)

        # Compute prediction error
        pred = model(image)
        loss = loss_fn(pred, graph)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            loss, current = loss.item(), i + 1
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

dataset = dl.LineGraphDataset('./train-ds')
optimizer = torch.optim.Adam(GraphModel.parameters(), lr=1e-5, weight_decay=1e-5)
Train(DataLoader(dataset, batch_size=None), GraphModel, AdjacencyCrossEntropy, optimizer)
