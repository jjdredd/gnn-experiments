#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle


def GeneratePlanarGraph(m, n):
    G = nx.barabasi_albert_graph(m, n)
    is_planar, embedding = nx.check_planarity(G, counterexample=True)
    while not is_planar:
        print(f'making graph {m}, {n} planar')
        edges = list(embedding.edges())[0]
        G.remove_edge(*edges)
        is_planar, embedding = nx.check_planarity(G, counterexample=True)

    return G

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
        G = GeneratePlanarGraph(v, e)
        e = v = 0

        plt.figure(figsize=(6, 6))
        nx.draw_planar(G, #pos=nx.planar_layout(G),
                with_labels=False,
                node_color='lightblue',
                node_size=0,
                width=2,
                font_weight='bold',
                edge_color='black'
                )
        plt.savefig(f'{destination}/graph_{i}.png')
        plt.clf()
        plt.close('all')

        with open(f'{destination}/graph_{i}.gpickle', 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    GenerateSamples('/tmp/graph_samples', 20, (6, 10), (2, 5))
