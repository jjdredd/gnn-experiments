#! /usr/bin/env python3

from sage.all import *
import random

train_set_size = 50000
test_set_size = 10000

# this function is not suitable: the graphs have self intersections

def GenerateSamples(destination: str, samples: int,
                    vertices: tuple[int, int], edges: tuple[int, int]):
    if edges[0] >= edges[1] or vertices[0] >= vertices[1] or edges[0] < 1:
        return

    e = 0
    v = 0

    for i in range(samples):
        while e == 0 or v == 0 or e < 1 or e >= v:
            e = random.randrange(edges[0], edges[1])
            v = random.randrange(vertices[0], vertices[1])
        print(v, e)
        G = graphs.RandomBarabasiAlbert(Integer(v), Integer(e))
        e = v = 0
        plot = G.plot(vertex_labels=False, vertex_size=Integer(10))
        plot.save(f'{destination}/graph_{i}.png')
        G.save(f'{destination}/graph_{i}')

if __name__ == '__main__':
    GenerateSamples('/tmp/graph_samples', 200, (5, 9), (2, 4))

