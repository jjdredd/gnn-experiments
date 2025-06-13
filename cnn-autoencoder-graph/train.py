#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.io import decode_image
from time import perf_counter_ns

import model
import data_loader as dl
import test

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
else:
    device = torch.device("cpu")

print(f"Using {device} device")


# GraphModel = model.CnnGraphEncoder().to(device)
GraphModel = model.CnnGraphEncoderDeconv().to(device)
print(GraphModel)

StdCrossEntropyLoss = nn.CrossEntropyLoss()
def AdjacencyCrossEntropy(prediction, ground_truth):
    return StdCrossEntropyLoss(prediction.flatten(start_dim=1), ground_truth)

# Support only square images for now
def Train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader)
    model.train()
    for i, data in enumerate(dataloader):
        image = data['image'].to(device)
        graph = data['graph'].to(device)

        # Compute prediction error
        pred = model(image)
        # reshape here because input has an additional dimension: channel
        loss = loss_fn(pred, graph)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            loss, current = loss.item(), i + 1
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

dataset = dl.LineGraphDataset('./train-32')
optimizer = torch.optim.Adam(GraphModel.parameters(), lr=1e-4, weight_decay=1e-6)

for data in DataLoader(dataset, batch_size=10):
    X = data['image']
    y = data['graph']
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

dataset_test = dl.LineGraphDataset('./test-ds')

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    t_start = perf_counter_ns()
    Train(DataLoader(dataset, batch_size=32), GraphModel, AdjacencyCrossEntropy, optimizer)
    test.Test(DataLoader(dataset_test, batch_size=32), GraphModel, AdjacencyCrossEntropy)
    t_stop = perf_counter_ns()
    print(f"Epoch {t+1} training finished in {t_stop - t_start} ns")
print("Done!")
