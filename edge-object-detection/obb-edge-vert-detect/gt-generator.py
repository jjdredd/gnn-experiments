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
    def __init__(self, num_pts: int, min_edge_size: float):
        self.min_edge_size = min_edge_size
        self.num_pts = num_pts
        self.margin = 0.1
        self.tolerance = 10**(-2)
        self.envelope = shp.Polygon(((0., 0.),
                                     (0., 1.),
                                     (1., 1.),
                                     (1., 0.),
                                     (0., 0.)))
        self.graph_vertices = []
        self.graph_edges = [[], []]

    def getGraphVertexIndex(v):
        for 

    def addGraphVertex(v):
        index = self.getGraphVertexIndex(v)
        if index is not None:
            return index
        self.graph_vertices.append(v)
        return len(self.graph_vertices) - 1

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
        for line in voronoi_diagram.geoms:
            if line.length < self.min_edge_size:
                continue
            line.coords



if __name__ == '__main__':
    generator = VoronoiGraphGenerator(10, 0.1)
    generator.generateVoronoiDiagram()
    print('Done!')
