#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt

# Create an empty graph
G = nx.Graph()

# Define positions for each vertex of the square
pos = {
    'A': (0, 0),
    'B': (1, 0),
    'C': (1, 1),
    'D': (0, 1)
}

# Add nodes to the graph (using positions from our dictionary)
for node in pos:
    G.add_node(node)

# Add edges for the square (the boundary of the square)
G.add_edge('A', 'B')
G.add_edge('B', 'C')
G.add_edge('C', 'D')
G.add_edge('D', 'A')

# Add diagonal edges: AC and BD
G.add_edge('A', 'C')
G.add_edge('B', 'D')

F = nx.barabasi_albert_graph(8, 4)

# Plot the graph using matplotlib
is_planar, embedding = nx.check_planarity(F, counterexample=True)
if is_planar:
    print("The F graph is planar.")
    plt.figure(figsize=(6, 6))
    nx.draw(F, pos=nx.planar_layout(F),
            with_labels=True,
            node_color='lightblue',
            node_size=800,
            font_weight='bold',
            edge_color='gray'
            )
else:
    print("The F graph is non-planar.")
    print(embedding)
    plt.figure(figsize=(6, 6))
    nx.draw(F,
            with_labels=True,
            node_color='lightblue',
            node_size=800,
            font_weight='bold',
            edge_color='gray'
            )
    plt.savefig("non-planar.png")
    plt.figure(figsize=(6, 6))
    nx.draw(embedding,
            with_labels=True,
            node_color='lightblue',
            node_size=800,
            font_weight='bold',
            edge_color='gray'
            )
    plt.savefig("embedding.png")

