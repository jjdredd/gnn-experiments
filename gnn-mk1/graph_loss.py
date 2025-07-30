#! /usr/bin/env python3

import torch
from torch import nn

# no vector may have a zero coordinate!
def EdgeMatrix2D(v_1, v_2):
    return torch.diag(
        torch.reciprocal(
            torch.matmul(torch.mul(v_1, v_2), torch.diag(torch.tensor([1.0, -1.0])))
        ))

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



class GraphLossRes(nn.Module):
    def __init__(self, epsilon=10**(-4)):
        super(GraphLossRes, self).__init__()
        self.epsilon = epsilon

    # this only accepts batched input
    def forward(self, vertices, edges, edge_features, edge_matrices):
        """
        Function to calculate the loss.
        Arguments:
        vertices: vertex features (tensor, shape (n, m) )
        edges: vertex indices forming edges (tensor, shape (2, n))
        edge_features: edge probability (tensor, shape (n))
        targets: matrices representing the ground truth edges
        """
        edge_matrices.requires_grad_(requires_grad=False)
        filtered_edges = FilterRepeatedEdges(edges)
        v_1 = vertices[filtered_edges[0]]
        v_2 = vertices[filtered_edges[1]]
        linear_product = torch.einsum('...kij,...mj->...kmi', edge_matrices, v_2)
        bilinear_form = torch.einsum('...kmi,...mi->...km', linear_product, v_1)
        # consider using torch.pow
        regularized_reciprocal = torch.reciprocal(torch.square(bilinear_form) + self.epsilon)
        return -torch.sum(torch.einsum('...km,...m->...k', regularized_reciprocal, edge_features))


class GraphLossSs(nn.Module):
    def __init__(self):
        super(GraphLossSs, self).__init__()

    # this only accepts batched input
    def forward(self, vertices, edges, edge_features, edge_matrices):
        """
        Function to calculate the loss.
        Arguments:
        vertices: vertex features (tensor, shape (n, m) )
        edges: vertex indices forming edges (tensor, shape (2, n))
        edge_features: edge probability (tensor, shape (n))
        targets: matrices representing the ground truth edges
        """
        edge_matrices.requires_grad_(requires_grad=False)
        filtered_edges = FilterRepeatedEdges(edges)
        v_1 = vertices[filtered_edges[0]]
        v_2 = vertices[filtered_edges[1]]
        linear_product = torch.einsum('...kij,...mj->...kmi', edge_matrices, v_2)
        bilinear_form = torch.einsum('...kmi,...mi->...km', linear_product, v_1)
        return -torch.sum(torch.square(torch.diagonal(bilinear_form)))


# XXX FIXME:
# For these losses (where we multiply each expression by the edge weights (probabilities))
# we can't just use probabilites, because the optimizer will either
# 1. set all the probabilties to zero in case of the sum of squares loss
# 2. create extra edges in case of the sum of reciprocal squares loss
# To combat this we can add an l2 norm of the edge probabilites to the loss
# with a sign that depends on case above (either case 1 or 2)

# To fix this, use a saddle potential (x - 0.5) * (y - 0.5)
