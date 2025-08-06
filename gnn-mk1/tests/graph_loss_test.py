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

    # GraphLossResidualSum
    # def test_GraphLossResidualSum_MinVal(self):
    #     loss = gl.GraphLossResidualSum(epsilon=10**(-3))
    #     vertices = torch.stack([torch.randn(2) for _ in range(2)], dim=0)
    #     edges = torch.tensor([[0, 1], [1, 0]])
    #     edge_features = torch.tensor([1.0]) # or edge weights in this case
    #     edge_matrices = gl.EdgeMatrices(vertices, edges)
    #     fake_vertices = vertices + 0.1 * torch.stack([torch.randn(2) for _ in range(2)], dim=0)
    #     gt_loss_val = loss.forward(fake_vertices, edges, edge_features, edge_matrices)
    #     print('test_GraphLossResidualSum_MinVal:gt_loss_val', gt_loss_val)

    # def test_GraphLossResidualSum_MinVal3(self):
    #     loss = gl.GraphLossResidualSum(epsilon=10**(-3))
    #     vertices = torch.stack([torch.randn(2) for _ in range(3)], dim=0)
    #     edges = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 2]])
    #     edge_features = torch.tensor([1.0, 1.0]) # or edge weights in this case
    #     edge_matrices = gl.EdgeMatrices(vertices, edges)
    #     fake_vertices = vertices # + 0.1 * torch.stack([torch.randn(2) for _ in range(3)], dim=0)
    #     fake_edges = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]])
    #     gt_loss_val = loss.forward(fake_vertices, fake_edges, edge_features, edge_matrices)
    #     print('test_GraphLossResidualSum_MinVal3:gt_loss_val', gt_loss_val)

    # GraphLossSumSquareBilinear

    # Create the following tests:
    # loss value increase due to:
    # 1. vertex moving
    # 2. confidence changing
    # 3. extra edge
    # 4. missing edge


class TestGraphLossResidualSum(unittest.TestCase):
    # may want to use setUpClass() instead, see docs
    def setUp(self):
        self.loss = gl.GraphLossResidualSum()
        self.vertices = torch.tensor([[2.0, 1.0],
                                      [2.0, 2.0],
                                      [1.0, 2.0]])
        self.edges = torch.tensor([[0, 1, 2, 0], [1, 0, 0, 2]])
        self.edge_features = torch.tensor([1.0, 1.0]) # or edge weights in this case
        self.edge_matrices = gl.EdgeMatrices(self.vertices, self.edges)
        self.base_loss = self.loss(self.vertices, self.edges, self.edge_features, self.edge_matrices)
        print('base loss: ', self.base_loss)

    def test_DisplacedVertex(self):
        fake_vertices = torch.clone(self.vertices)
        fake_vertices[1] += torch.tensor([0.01, 0.2])
        new_loss = self.loss(fake_vertices, self.edges, self.edge_features, self.edge_matrices)
        self.assertLess(self.base_loss.item(), new_loss.item())

    def test_WrongWeight(self):
        fake_edge_features = torch.tensor([1.0, 0.2])
        new_loss = self.loss(self.vertices, self.edges, fake_edge_features, self.edge_matrices)
        self.assertLess(self.base_loss.item(), new_loss.item())
        fake_edge_features = torch.tensor([0.2, 1.0])
        new_loss = self.loss(self.vertices, self.edges, fake_edge_features, self.edge_matrices)
        self.assertLess(self.base_loss.item(), new_loss.item())

    def test_MissingEdge(self):
        fake_edges = torch.tensor([[0, 1], [1, 0]])
        fake_edge_features = torch.tensor([1.0])
        new_loss = self.loss(self.vertices, fake_edges, self.edge_features, self.edge_matrices)
        self.assertLess(self.base_loss.item(), new_loss.item())
        fake_edge_features = torch.tensor([0.2])
        new_loss = self.loss(self.vertices, fake_edges, fake_edge_features, self.edge_matrices)
        self.assertLess(self.base_loss.item(), new_loss.item())

    def test_ExtraEdge(self):
        print('extra edge')
        fake_edges = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 0, 0, 2, 2, 1]])
        fake_edge_features = torch.tensor([1.0, 1.0, 1.0])
        new_loss = self.loss(self.vertices, fake_edges, fake_edge_features, self.edge_matrices)
        self.assertLess(self.base_loss.item(), new_loss.item())
        fake_edge_features = torch.tensor([1.0, 1.0, 0.0])
        new_loss = self.loss(self.vertices, fake_edges, fake_edge_features, self.edge_matrices)
        self.assertAlmostEqual(self.base_loss.item(), new_loss.item())


if __name__ == "__main__":
    unittest.main()
