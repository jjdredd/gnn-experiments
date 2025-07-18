#! /usr/bin/env python3

import torch
from torch import nn

# no vector may have a zero coordinate!
def EdgeMatrix2D(v_1, v_2):
    return torch.reciprocal(torch.matmul(torch.mul(v_1, v_2), torch.diag(torch.tensor([1.0, -1.0]))))

def FilterRepeatedEdges(edges):
    """
    Returns edges filtered, without duplicates
    """
    result = edges[:, 0].unsqueeze(dim=1)
    edge_set = set()
    edge_set.add((result[0, 0].item(), result[1, 0].item()))
    for n in range(1, edges.shape[1]):
        reverse_edge = (edges[1, n].item(), edges[0, n].item())
        normal_edge = (edges[0, n].item(), edges[1, n].item())
        if reverse_edge in edge_set or normal_edge in edge_set:
            continue
        else:
            edge_set.add(normal_edge)
            result = torch.cat((result, edges[:, n].unsqueeze(dim=1)), dim=1)
    return result
        

def EdgeMatrices(vertices, edges):
    """
    Compute a loss function matrix for each edge.
    vertices: vertex features (tensor, shape (n, m) )
    edges: vertex indices forming edges (tensor, shape (2, n))
    must return a tensor of matrices (shape (n, m, m))
    """
    filtered_edges = FilterRepeatedEdges(edges)
    V_1 = vertices[filtered_edges[0]]
    V_2 = vertices[filtered_edges[1]]
    matrix_list = []
    for v_1, v_2 in zip(V_1, V_2):
        matrix_list.append(EdgeMatrix2D(v_1, v_2))
    return torch.stack(matrix_list, dim=0)


class GraphLoss(nn.Module):
    def __init__(self, singularity_cutoff=1000):
        super(CustomLoss, self).__init__()
        self.singularity_cutoff = singularity_cutoff

    def forward(self, vertices, edges, edge_features, edge_matrices):
        """
        Function to calculate the loss.
        Arguments:
        vertices: vertex features (tensor, shape (n, m) )
        edges: vertex indices forming edges (tensor, shape (2, n))
        edge_features: edge probability (tensor, shape (n))
        targets: matrices representing the ground truth edges
        """
        filtered_edges = FilterRepeatedEdges(edges)
        V_1 = vertices[filtered_edges[0]]
        V_2 = vertices[filtered_edges[1]]



