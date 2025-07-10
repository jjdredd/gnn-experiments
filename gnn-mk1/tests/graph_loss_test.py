#! /usr/bin/env python3

import unittest
import torch
import graph_loss as gl

class TestEdgeMatrices(unittest.TestCase):
    def test_SingleEdge(self):
        vertices = torch.stack([torch.randn(2) for _ in range(2)], dim=0)
        edges = torch.tensor([[0, 1], [1, 0]])
        matrices = gl.EdgeMatrices(vertices, edges)
        loss = torch.matmul(vertices[0], torch.mul(matrices[0], vertices[1]))
        # print(loss, matrices)
        self.assertTrue(abs(loss) < 10**(-5))

    def test_FilterRepeatedEdges(self):
        edges = torch.tensor([[0, 1, 1, 2, 4, 5, 3, 8, 10],
                              [1, 2, 3, 1, 10, 7, 1, 6, 4]])
        edges_expected = torch.tensor([[0, 1, 1, 4, 5, 8],
                                       [1, 2, 3, 10, 7, 6]])
        edges_processed = gl.FilterRepeatedEdges(edges)
        self.assertTrue(torch.all(torch.isclose(edges_processed, edges_expected)).item())

    def test_GraphLoss_MinVal(self):
        loss = gl.GraphLoss(singularity_cutoff=100)
        vertices = torch.stack([torch.randn(2) for _ in range(2)], dim=0)
        edges = torch.tensor([[0, 1], [1, 0]])
        edge_features = torch.tensor([1.0, 1.0]) # or edge weights in this case
        expected_min_val = 
        edge_matrices = EdgeMatrices(vertices, edges)
        gt_loss_val = loss.forward(vertices, edges, edge_features, edge_matrices)

    def test_GraphLoss
        # manipulate the vertices/edges and make sure the new loss is greater than the gt_loss_val

if __name__ == "__main__":
    unittest.main()
