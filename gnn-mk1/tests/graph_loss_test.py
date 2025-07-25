#! /usr/bin/env python3

import unittest
import torch
import graph_loss as gl

class TestEdgeMatrices(unittest.TestCase):
    def test_SingleEdge(self):
        vertices = torch.stack([torch.randn(2) for _ in range(2)], dim=0)
        edges = torch.tensor([[0, 1], [1, 0]])
        matrices = gl.EdgeMatrices(vertices, edges)
        loss = torch.matmul(vertices[0], torch.matmul(matrices[0], vertices[1]))
        # print(loss, matrices)
        self.assertTrue(abs(loss) < 10**(-5))

    def test_FilterRepeatedEdges(self):
        edges = torch.tensor([[0, 1, 1, 2, 4, 5, 3, 8, 10],
                              [1, 2, 3, 1, 10, 7, 1, 6, 4]])
        edges_expected = torch.tensor([[0, 1, 1, 4, 5, 8],
                                       [1, 2, 3, 10, 7, 6]])
        edges_processed = gl.FilterRepeatedEdges(edges)
        self.assertTrue(torch.all(torch.isclose(edges_processed, edges_expected)).item())

    # GraphLossRes
    def test_GraphLossRes_MinVal(self):
        loss = gl.GraphLossRes(epsilon=10**(-3))
        vertices = torch.stack([torch.randn(2) for _ in range(2)], dim=0)
        edges = torch.tensor([[0, 1], [1, 0]])
        edge_features = torch.tensor([1.0]) # or edge weights in this case
        edge_matrices = gl.EdgeMatrices(vertices, edges)
        fake_vertices = vertices + 0.1 * torch.stack([torch.randn(2) for _ in range(2)], dim=0)
        gt_loss_val = loss.forward(fake_vertices, edges, edge_features, edge_matrices)
        print('test_GraphLossRes_MinVal:gt_loss_val', gt_loss_val)

    def test_GraphLossRes_MinVal3(self):
        loss = gl.GraphLossRes(epsilon=10**(-3))
        vertices = torch.stack([torch.randn(2) for _ in range(3)], dim=0)
        edges = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 2]])
        edge_features = torch.tensor([1.0, 1.0]) # or edge weights in this case
        edge_matrices = gl.EdgeMatrices(vertices, edges)
        fake_vertices = vertices # + 0.1 * torch.stack([torch.randn(2) for _ in range(3)], dim=0)
        fake_edges = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
        gt_loss_val = loss.forward(fake_vertices, fake_edges, edge_features, edge_matrices)
        print('test_GraphLossRes_MinVal3:gt_loss_val', gt_loss_val)

    # GraphLossSs
    def test_GraphLossSs_MinVal(self):
        loss = gl.GraphLossSs()
        vertices = torch.stack([torch.randn(2) for _ in range(2)], dim=0)
        edges = torch.tensor([[0, 1], [1, 0]])
        edge_features = torch.tensor([1.0]) # or edge weights in this case
        edge_matrices = gl.EdgeMatrices(vertices, edges)
        fake_vertices = vertices # + 0.1 * torch.stack([torch.randn(2) for _ in range(2)], dim=0)
        gt_loss_val = loss.forward(fake_vertices, edges, edge_features, edge_matrices)
        print('test_GraphLossSs_MinVal:gt_loss_val', gt_loss_val)

    def test_GraphLossSs_MinVal3(self):
        loss = gl.GraphLossSs()
        vertices = torch.stack([torch.randn(2) for _ in range(3)], dim=0)
        edges = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 2]])
        edge_features = torch.tensor([1.0, 1.0]) # or edge weights in this case
        edge_matrices = gl.EdgeMatrices(vertices, edges)
        print('edge_matrices', edge_matrices)
        fake_vertices = vertices # + 0.1 * torch.stack([torch.randn(2) for _ in range(3)], dim=0)
        fake_edges = edges # torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
        gt_loss_val = loss.forward(fake_vertices, fake_edges, edge_features, edge_matrices)
        print('test_GraphLossSs_MinVal3:gt_loss_val', gt_loss_val)


    def test_GraphLossRes(self):
        # manipulate the vertices/edges and make sure the new loss is greater than the gt_loss_val
        pass

if __name__ == "__main__":
    unittest.main()
