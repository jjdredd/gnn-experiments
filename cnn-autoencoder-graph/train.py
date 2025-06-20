#!/usr/bin/env python3

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, GaussianBlur
from torchvision.io import decode_image
from torchvision.utils import save_image
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
GraphModel = model.CnnGraphEncoder_Ntc_Wes().to(device)
print(GraphModel)

upsampling = nn.Upsample(scale_factor=3, mode='nearest')
StdCrossEntropyLoss = nn.CrossEntropyLoss()
def AdjacencyCrossEntropy(prediction, ground_truth):
    return StdCrossEntropyLoss(prediction, ground_truth)

# Support only square images for now
def Train(dataloader, model, loss_fn, optimizer, epoch):
    sigma = 8 / (epoch + 1)
    blur = GaussianBlur(kernel_size=(9, 9), sigma=(sigma, sigma))
    size = len(dataloader)
    model.train()
    for i, data in enumerate(dataloader):
        image = data['image'].to(device)
        graph = data['graph'].to(device)

        # if epoch < 20:
        #     graph = blur(graph)

        # Compute prediction error
        pred = model(image)
        save_image(image[0], 'input.png')
        save_image(pred[0], 'pred.png')
        # print('pred.shape', pred.shape)
        # print('graph.shape', graph.shape)
        # reshape here because input has an additional dimension: channel
        loss = loss_fn(pred, graph)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            loss, current = loss.item(), i + 1
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # save_image(image[0], f'pics/input_{i}.png')
            save_image(torch.cat((pred[0], graph[0]), 1), f'pics/pred_gt_{i}.png')
            # save_image(, f'pics/gt_{i}.png')


dataset = dl.LineGraphDataset('./train-32')
optimizer = torch.optim.Adam(GraphModel.parameters(), lr=1e-4, weight_decay=1e-5)

for data in DataLoader(dataset, batch_size=10):
    X = data['image']
    y = data['graph']
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

dataset_test = dl.LineGraphDataset('./test-ds')

epochs = 700
batch_size = 64
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    t_start = perf_counter_ns()
    Train(DataLoader(dataset, batch_size=batch_size), GraphModel, AdjacencyCrossEntropy, optimizer, t)
    test.Test(DataLoader(dataset_test, batch_size=batch_size), GraphModel, AdjacencyCrossEntropy)
    t_stop = perf_counter_ns()
    print(f"Epoch {t+1} training finished in {t_stop - t_start} ns")
print("Done!")

torch.save(GraphModel.state_dict(), 'GraphModel.2.weights.saved')

# Make regularization: penalize the number of vertices/edges
