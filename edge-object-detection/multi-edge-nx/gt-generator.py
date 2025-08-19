#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import random
import yaml

import networkx as nx
import matplotlib.pyplot as plt
import random
import pickle


ImageSize = 100
NumTrainSamples = 30000
NumValSamples = 1000

EdgeThickness = 2


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


class YoloDataGenerator():
    def __init__(self,
                 output_directory: str,
                 nodes: list[float],
                 edges: list[float],
                 training_samples=80000,
                 validation_samples=1000,
                 min_line_dim=5):
        self.output_directory = output_directory
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.min_line_dim = min_line_dim
        self.image_size = ImageSize
        self.nodes = nodes
        self.edges = edges

    @staticmethod
    def PixelCompare(x, y):
        return abs(x - y) < 1.0

    @staticmethod
    def CalculateCenter(p_1, p_2):
        return ((p_1[0] + p_2[0]) / 2.0, (p_1[1] + p_2[1]) / 2.0)

    @staticmethod
    def EdgeBoungdingBox(p_1, p_2):
        return (abs(p_2[0] - p_1[0]), abs(p_2[1] - p_1[1]))

    def ClassifyEdge(self, p_1, p_2) -> int:
        """
        Classify edge. The classes are:
        0. 0-th and 2-nd quadrant
        1. 1-st and 3-rd quadrant
        2. horizontal
        3. vertical
        """
        if YoloDataGenerator.PixelCompare(p_1[1], p_2[1]):
            return 2
        elif YoloDataGenerator.PixelCompare(p_1[0], p_2[0]):
            return 3
        elif (p_2[0] - p_1[0]) * (p_2[1] - p_1[1]) > 0:
            return 0
        else:
            return 1

    def NormalizeCoordinates(self, p):
        return (p[0] / self.image_size, p[1] / self.image_size)

    def GeneratePlanarGraph(self, nodes, edges):
        G = nx.barabasi_albert_graph(nodes, edges)
        is_planar, embedding = nx.check_planarity(G, counterexample=True)
        while not is_planar:
            edges = list(embedding.edges())[0]
            G.remove_edge(*edges)
            is_planar, embedding = nx.check_planarity(G, counterexample=True)
        return G

    def GenerateGraphRandom(self):
        edges, nodes = 0, 0
        while edges < 1 or nodes <= edges:
            edges = random.randrange(self.edges[0], self.edges[1])
            nodes = random.randrange(self.vertices[0], self.vertices[1])
        return self.GeneratePlanarGraph(nodes, edges)

    def RenderGraph(self, graph, file_path):
        plt.figure(figsize=(1, 1), dpi=self.image_size)
        nx.draw_networkx_edges(graph, #pos=nx.planar_layout(G),
                       with_labels=False,
                       node_color='black', # node_color='lightblue',
                       node_size=0,
                       width=EdgeThickness,
                       font_weight='bold',
                       edge_color='black')
        plt.savefig(file_path)
        plt.clf()
        plt.close('all')

    def RecordGraphEdges(self, graph, file_path):
        txt_file = open(f'{label_file_directory}/{base_name}.txt', 'w')
        
        pass

    def GenerateSample(self, base_name, directory_path, training):
        subdirectory = 'train' if training else 'val'
        label_file_directory = f'{directory_path}/labels/{subdirectory}'
        image_file_directory = f'{directory_path}/images/{subdirectory}'
        EnsureDirectoryExists(label_file_directory)
        EnsureDirectoryExists(image_file_directory)

        graph = self.GeneratePlanarGraph()
        self.RenderGraph(graph, f'{image_file_directory}/{base_name}.png')
        self.RecordGraphEdges(graph, f'{label_file_directory}/{base_name}.txt')

        # Define start and end points of the line
        start_point = (x_1, y_1)
        end_point = (x_2, y_2)
        image_start_point = (start_point[0], self.image_size - start_point[1] - 1)
        image_end_point = (end_point[0], self.image_size - end_point[1] - 1)
        edge_class = self.ClassifyEdge(start_point, end_point)
        edge_center = self.NormalizeCoordinates(YoloDataGenerator.CalculateCenter(image_start_point,
                                                                                  image_end_point))
        edge_bb = self.NormalizeCoordinates(YoloDataGenerator.EdgeBoungdingBox(image_start_point, 
                                                                               image_end_point))
        # need to normalize coordinates
        txt_file.write(f'{edge_class}\t{edge_center[0]}\t{edge_center[1]}\t{edge_bb[0]}\t{edge_bb[1]}\n')

        image = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)  # Example: black image
        # Define line color (BGR format)
        color = (255, 255, 255)
        # Define line thickness
        thickness = EdgeThickness
        # Draw the line
        cv2.line(image, image_start_point, image_end_point, color, thickness)
        cv2.imwrite(f'{image_file_directory}/{base_name}.png', image)

    def GenerateDataSets(self):
        EnsureDirectoryExists(self.output_directory)
        # generate the training data set
        for n in range(NumTrainSamples):
            self.GenerateSample(f'line_{n}', self.output_directory, True)
        # generate the 
        for n in range(NumValSamples):
            self.GenerateSample(f'line_{n}', self.output_directory, False)

        data_dict = {'train': 'images/train',
                     'val': 'images/val',
                     'names': {
                         0: 'posslant',
                         1: 'negslant',
                         2: 'horizontal',
                         3: 'vertical'}}
        with open(f'{self.output_directory}/data.yaml', 'w+') as yaml_config_file:
            yaml.dump(data_dict, yaml_config_file)

if __name__ == '__main__':
    # if len(sys.argv) < 2:
    #     print('Destination directory required')
    #     exit(-1)

    ydg = YoloDataGenerator('/mnt/tmpfs/yolo-training-data')
    ydg.GenerateDataSets()
    print('Done!')


# this training command is essential 
# training command
# yolo detect train data=./yolo-training-data/data.yaml model=yolo11s.pt augment=False epochs=16 pretrained=False hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 degrees=0.0 translate=0.0 scale=0.0 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.0 bgr=0.0 mosaic=0.0 mixup=0.0 cutmix=0.0 copy_paste=0.0 erasing=0.0
