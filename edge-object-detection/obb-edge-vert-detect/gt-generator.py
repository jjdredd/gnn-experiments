#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import yaml

import matplotlib.pyplot as plt
import shapely as shp

MinEdgeSize = 10                # pixels?

ImageSize = 416
EdgeThickness = 0.5

NodeNums = [8, 10]
EdgeNums = [1, 5]


def EnsureDirectoryExists(directory_path):
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            return True
        else:
            return True
    except OSError as error:
        print(f"Error creating directory {directory_path}: {error}")
        return False


class VoronoiGraphGenerator():
    def __init__(self, num_pts: int, min_edge_size: float,
                 decimate: int):
        self.min_edge_size = min_edge_size
        self.num_pts = num_pts
        self.decimate = decimate
        self.margin = 0.1
        self.tolerance = 10**(-2)
        self.envelope = shp.Polygon(((0., 0.),
                                     (0., 1.),
                                     (1., 1.),
                                     (1., 0.),
                                     (0., 0.)))
        self.graph_vertices = []
        self.graph_edges = [[], []]

    def getGraphVertexIndex(self, v):
        for n, vertex in enumerate(self.graph_vertices):
            if np.allclose(vertex, v):
                return n
        return None

    def addGraphVertex(self, v):
        index = self.getGraphVertexIndex(v)
        if index is not None:
            return index
        self.graph_vertices.append(v)
        return len(self.graph_vertices) - 1

    def addGraphEdge(self, edge):
        index_1 = self.addGraphVertex(edge.coords[0])
        index_2 = self.addGraphVertex(edge.coords[1])
        # forward
        self.graph_edges[0].append(index_1)
        self.graph_edges[1].append(index_2)
        # backward
        self.graph_edges[1].append(index_1)
        self.graph_edges[0].append(index_2)

    def filterShortEdges(self, edges):
        return [line for line in edges
                if line.length >= self.min_edge_size]

    def decimateEdges(self, edges):
        return [line for line in edges
                if np.random.randint(self.decimate) != 0]

    def linesToGraph(self, edges):
        for edge in edges:
            self.addGraphEdge(edge)

    def generateRandomPoints(self):
        points_list = np.random.uniform(self.margin, 1.0 - self.margin,
                                        size=(self.num_pts, 2))
        return shp.MultiPoint(points_list)

    def generateVoronoiDiagram(self):
        voronoi_diagram = shp.voronoi_polygons(self.generateRandomPoints(),
                                               tolerance=self.tolerance,
                                               extend_to=self.envelope,
                                               only_edges=True,
                                               ordered=False)
        edges = self.filterShortEdges(voronoi_diagram.geoms)
        edges = self.decimateEdges(edges)
        self.linesToGraph(edges)


class GraphRenderer():
    def __init__(self, image_size: int):
        self.image_size = image_size

    def RenderGraph(self):
        pass

    def AnnotateGraph(self):
        pass

    def RenderAnnotation(self, image):
        pass


if __name__ == '__main__':
    generator = VoronoiGraphGenerator(10, 0.02, 20)
    generator.generateVoronoiDiagram()
    print(generator.graph_vertices)
    print(generator.graph_edges)
    print('Done!')
