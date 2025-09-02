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


def NpArrayTotuple(v):
    return (int(v[0].item()), int(v[1].item()))


class VoronoiGraphGenerator():
    def __init__(self, num_pts: int, min_edge_size: float,
                 decimate: int):
        self.min_edge_size = min_edge_size
        self.num_pts = num_pts
        self.decimate = decimate
        self.margin = 0.1
        self.tolerance = 10**(-2)
        self.width_range = [0.01, 0.05]
        self.balance_range = [0.2, 0.8]
        self.envelope = shp.Polygon(((0., 0.),
                                     (0., 1.),
                                     (1., 1.),
                                     (1., 0.),
                                     (0., 0.)))
        self.graph_vertices = []
        self.graph_edges = [[], []]
        self.edge_width = []
        self.edge_balance = []

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
        index_1 = self.addGraphVertex(np.array(edge.coords[0]))
        index_2 = self.addGraphVertex(np.array(edge.coords[1]))
        width = np.random.uniform(self.width_range[0], self.width_range[1])
        balance = np.random.uniform(self.balance_range[0], self.balance_range[1])
        # forward
        self.graph_edges[0].append(index_1)
        self.graph_edges[1].append(index_2)
        self.edge_width.append(width)
        self.edge_balance.append(balance)
        # backward
        self.graph_edges[1].append(index_1)
        self.graph_edges[0].append(index_2)
        self.edge_width.append(width)
        self.edge_balance.append(1 - balance)

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
        edges = shp.intersection(voronoi_diagram.geoms, self.envelope)
        edges = self.filterShortEdges(edges)
        edges = self.decimateEdges(edges)
        self.linesToGraph(edges)


class GraphRenderer():
    def __init__(self, image_size: int):
        self.image_size = image_size
        self.vertex_box_size = 0.025
        self.margin_ratio = 0.1
        self.interior_image_size = (1 - 2 * self.margin_ratio) * self.image_size

    def relativeToFullImageCoord(self, p):
        margin_vector = np.array([self.margin_ratio] * 2)
        res = self.interior_image_size * p
        res += self.image_size * margin_vector
        res[1] = self.image_size - res[1]
        return res

    def relativeToRelativeImage(self, p):
        return self.relativeToFullImageCoord(p) / self.image_size

    @staticmethod
    def normalUnitVec(p):
        if np.allclose(p, np.zeros(2)):
            return p
        normal_vec = np.array([p[1], -p[0]])
        return normal_vec / np.linalg.norm(normal_vec)

    @staticmethod
    def edgeVec(p):
        return p[1] - p[0]

    @staticmethod
    def edgeTransverseUnitVec(p):
        return GraphRenderer.normalUnitVec(GraphRenderer.edgeVec(p))

    @staticmethod
    def edgePolygon(edge, width: float, balance: float):
        edge_vec = GraphRenderer.edgeVec(edge)
        edge_unit_normal = GraphRenderer.edgeTransverseUnitVec(edge)
        p_1 = edge[0] + width * balance * edge_unit_normal
        p_2 = p_1 + edge_vec
        p_3 = p_2 - width * edge_unit_normal
        p_4 = p_3 - edge_vec
        return [p_1, p_2, p_3, p_4]

    def vertexPolygon(self, p):
        e_1 = np.array([1.0, 0.0])
        e_2 = np.array([0.0, 1.0])
        p_1 = p - 0.5 * self.vertex_box_size * (e_1 + e_2)
        p_2 = p_1 + self.vertex_box_size * e_1
        p_3 = p_2 + self.vertex_box_size * e_2
        p_4 = p_3 - self.vertex_box_size * e_1
        return [p_1, p_2, p_3, p_4]

    def RenderGraph(self, image_path, graph: VoronoiGraphGenerator):
        render_vert = False
        written_edges = set()
        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        color = (255, 255, 255)
        for n, edge_index in enumerate(zip(graph.graph_edges[0], graph.graph_edges[1])):
            edge_index_tuple = (edge_index[0], edge_index[1])
            if edge_index_tuple in written_edges:
                continue
            written_edges.add(edge_index_tuple)
            edge_index_tuple_r = (edge_index[1], edge_index[0])
            written_edges.add(edge_index_tuple_r)
            edge = [graph.graph_vertices[edge_index[0]],
                    graph.graph_vertices[edge_index[1]]]
            polygon = self.edgePolygon(edge, graph.edge_width[n],
                                       graph.edge_balance[n])
            poly_img = np.array([NpArrayTotuple(self.relativeToFullImageCoord(v))
                                 for v in polygon])
            cv2.fillConvexPoly(image, poly_img, color)

        if not render_vert:
            cv2.imwrite(image_path, image)
            return

        color = (0, 0, 255)
        for v in graph.graph_vertices:
            polygon = self.vertexPolygon(v)
            poly_img = np.array([NpArrayTotuple(self.relativeToFullImageCoord(v))
                                 for v in polygon])
            cv2.fillConvexPoly(image, poly_img, color)

        cv2.imwrite(image_path, image)

    # 0 - edge/wall
    # 1 - vertex
    def AnnotateGraph(self, txt_path, graph: VoronoiGraphGenerator):
        written_edges = set()
        txt_file = open(txt_path, 'w')
        for n, edge_index in enumerate(zip(graph.graph_edges[0], graph.graph_edges[1])):
            edge_index_tuple = (edge_index[0], edge_index[1])
            if edge_index_tuple in written_edges:
                continue
            written_edges.add(edge_index_tuple)
            edge = [graph.graph_vertices[edge_index[0]],
                    graph.graph_vertices[edge_index[1]]]
            polygon = self.edgePolygon(edge, graph.edge_width[n],
                                       graph.edge_balance[n])
            txt_file.write((f'0\t{polygon[0][0]}\t{polygon[0][1]}'
                            f'\t{polygon[1][0]}\t{polygon[1][1]}'
                            f'\t{polygon[2][0]}\t{polygon[2][1]}'
                            f'\t{polygon[3][0]}\t{polygon[3][1]}\n'))

        for vertex in graph.graph_vertices:
            polygon = self.vertexPolygon(vertex)
            txt_file.write((f'1\t{polygon[0][0]}\t{polygon[0][1]}'
                            f'\t{polygon[1][0]}\t{polygon[1][1]}'
                            f'\t{polygon[2][0]}\t{polygon[2][1]}'
                            f'\t{polygon[3][0]}\t{polygon[3][1]}\n'))

    def RenderAnnotation(self, image, annotation):
        pass


if __name__ == '__main__':
    generator = VoronoiGraphGenerator(14, 0.06, 10)
    generator.generateVoronoiDiagram()
    renderer = GraphRenderer(416)
    renderer.RenderGraph('./image.png', generator)
    renderer.AnnotateGraph('./annotation.txt', generator)
    print('Done!')
