#!/usr/bin/env python3

import torch
from torch import nn
from torchvision.transforms import ToTensor
from torchvision.io import decode_image
from time import perf_counter_ns

import model
import data_loader as dl

# Support only square images for now

def TensorToEdges(adjacency_tensor, image_shape):
    edge_thres = 0.75
    edges = []
    for i in range(adjacency_tensor.shape[0]):
        for j in range(adjacency_tensor.shape[1]):
            if adjacency_tensor[i, j] < edge_thres:
                continue
            start_coordinates = LineGraphDataset.LatentToImageIndices(i)
            end_coordinates = LineGraphDataset.LatentToImageIndices(j)
            edges.append((start_coordinates, end_coordinates))

    return edges
                

if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
else:
    device = torch.device("cpu")

print(f"Using {device} device")


GraphModel = model.CnnGraphEncoder().to(device)
print(model)

if False:
    GraphModel.eval()
    dataset = dl.LineGraphDataset('./test-ds')
    with torch.no_grad():
        image = dataset[0]['image']
        gt_graph_tensor = dataset[0]['graph']
        print('Input image shape ', image.shape)
        print('GT tensor shape ', gt_graph_tensor.shape)
        output_graph = GraphModel(image.to(device))
        print('output graph tensor shape ', output_graph.shape)

def TestSingle(image_path):
    GraphModel.eval()
    image = decode_image(img_path)
    image = image[0].to(torch.float32)
    output_adjacency = GraphModel(image.to(device))
    print('Edges in the image:')
    for edge in TensorToEdges(output_adjacency, image.shape):
        print(edge[0], ' -> ', edge[1])

def test():
    GraphModel.eval()
    dataset = dl.LineGraphDataset('./test-ds')
    with torch.no_grad():
        for data in dataset:
            image = data['image']


def Test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            image = data['image'].to(device)
            graph = data['graph'].to(device)

            # Compute prediction error
            pred = model(image)
            # reshape here because input has an additional dimension: channel
            test_loss += loss_fn(pred, graph).item()
    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

