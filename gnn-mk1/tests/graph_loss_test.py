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
        print(loss, matrices)
        self.assertTrue(abs(loss) < 10**(-5))

    def test_MultipleEdge(self):
        # todo
        pass


if __name__ == "__main__":
    unittest.main()
