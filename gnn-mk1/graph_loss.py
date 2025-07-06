#! /usr/bin/env python3

import torch
from torch import nn

# no vector may have a zero coordinate!
def EdgeMatrix2D(v_1, v_2):
    return = torch.reciprocal(torch.matmul(torch.mul(v_1, v_2), torch.diag(torch.tensor([1, -1]))))
    

def EdgeMatrices(vertices, edges):
    """
    Compute a loss function matrix for each edge.
    """
    edge_matrices = []
    for n in range(edges.shape[1]):
        edge_matrices.append(EdgeMatrix2D(edges[0][n]))


class GraphLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, vertices, edges, edge_features, targets):
        """
        Function to calculate the loss.
        Arguments:
        vertices: vertex features (tensor, shape (n, m) )
        edges: vertex indices forming edges (tensor, shape (2, n))
        edge_features: edge probability (tensor, shape (n))
        targets: matrices representing the ground truth edges
        """
        pass


