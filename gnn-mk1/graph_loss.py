#! /usr/bin/env python3

import torch
from torch import nn

# no vector may have a zero coordinate!
def EdgeMatrix2D(v_1, v_2):
    return torch.reciprocal(torch.matmul(torch.mul(v_1, v_2), torch.diag(torch.tensor([1.0, -1.0]))))
    

def EdgeMatrices(vertices, edges):
    """
    Compute a loss function matrix for each edge.
    vertices: vertex features (tensor, shape (n, m) )
    edges: vertex indices forming edges (tensor, shape (2, n))
    must return a tensor of matrices (shape (n, m, m))
    """
    V_1 = vertices[edges[0]]
    V_2 = vertices[edges[1]]
    # XXX FIXME
    # filter out the repeating edges!!
    matrix_list = []
    for v_1, v_2 in zip(V_1, V_2):
        matrix_list.append(EdgeMatrix2D(v_1, v_2))
    return torch.stack(matrix_list, dim=0)


class GraphLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, vertices, edges, edge_features, edge_matrices):
        """
        Function to calculate the loss.
        Arguments:
        vertices: vertex features (tensor, shape (n, m) )
        edges: vertex indices forming edges (tensor, shape (2, n))
        edge_features: edge probability (tensor, shape (n))
        targets: matrices representing the ground truth edges
        """
        V_1 = vertices[edges[0]]
        V_2 = vertices[edges[1]]



