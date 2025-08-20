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

ImageSize = 400
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


class YoloDataGenerator():
    def __init__(self,
                 output_directory: str,
                 training_samples=20000,
                 validation_samples=500):
        self.output_directory = output_directory
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.image_size = ImageSize
        self.edge_nums = EdgeNums
        self.node_nums = NodeNums
        self.eps = 10**(-3)
        self.min_bbox = 0.01

    def FuzzyCompare(self, x, y):
        return abs(x - y) < self.eps

    @staticmethod
    def CalculateCenter(p_1, p_2):
        return [(p_1[0] + p_2[0]) / 2.0, (p_1[1] + p_2[1]) / 2.0]

    @staticmethod
    def EdgeBoungdingBox(p_1, p_2):
        return [abs(p_2[0] - p_1[0]), abs(p_2[1] - p_1[1])]

    def EnsureNonZeroBoundingBox(self, bbox):
        bbox[0] = max(bbox[0], self.min_bbox)
        bbox[1] = max(bbox[1], self.min_bbox)

    def ClassifyEdge(self, p_1, p_2) -> int:
        """
        Classify edge. The classes are:
        0. 0-th and 2-nd quadrant
        1. 1-st and 3-rd quadrant
        2. horizontal
        3. vertical
        """
        if self.FuzzyCompare(p_1[1], p_2[1]):
            return 2
        elif self.FuzzyCompare(p_1[0], p_2[0]):
            return 3
        elif (p_2[0] - p_1[0]) * (p_2[1] - p_1[1]) > 0:
            return 0
        else:
            return 1

    @staticmethod
    def ToYoloCoordinates(p):
        return [p[0], 1 - p[1]]

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
            edges = random.randrange(self.edge_nums[0], self.edge_nums[1])
            nodes = random.randrange(self.node_nums[0], self.node_nums[1])
        return self.GeneratePlanarGraph(nodes, edges)

    def RenderGraph(self, graph, pos, file_path):
        plt.figure(figsize=(1, 1), dpi=self.image_size)
        nx.draw_networkx_edges(graph, pos=pos,
                               # with_labels=False, node_color='lightblue', font_weight='bold',
                               node_size=0,
                               width=EdgeThickness,
                               edge_color='black')
        plt.box(False)
        fig = plt.gcf()
        fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        ax = plt.gca()
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.savefig(file_path)
        plt.clf()
        plt.close('all')

    def RecordGraphEdges(self, graph, pos, file_path):
        txt_file = open(file_path, 'w')
        for edge in graph.edges:
            start_point = pos[edge[0]]
            end_point = pos[edge[1]]
            image_start_point = YoloDataGenerator.ToYoloCoordinates(start_point)
            image_end_point = YoloDataGenerator.ToYoloCoordinates(end_point)
            # the edge coordinates are suposed to be in range [0; 1]
            # according to the way 'pos' was calculated in self.GenerateSample()
            edge_class = self.ClassifyEdge(start_point, end_point)
            edge_center = YoloDataGenerator.CalculateCenter(image_start_point,
                                                            image_end_point)
            edge_bb = YoloDataGenerator.EdgeBoungdingBox(image_start_point,
                                                         image_end_point)
            # trick to prevent zero area bounding boxes
            # Make sure that horizontal and vertical bounding boxes are large enough
            # The exact corners are less important in this case because we don't use
            # corners for edge endpoints for horizontal/vertical edges
            self.EnsureNonZeroBoundingBox(edge_bb)
            txt_file.write(f'{edge_class}\t{edge_center[0]}\t{edge_center[1]}')
            txt_file.write(f'\t{edge_bb[0]}\t{edge_bb[1]}\n')

    def GenerateSample(self, base_name, directory_path, training):
        subdirectory = 'train' if training else 'val'
        label_file_directory = f'{directory_path}/labels/{subdirectory}'
        image_file_directory = f'{directory_path}/images/{subdirectory}'
        EnsureDirectoryExists(label_file_directory)
        EnsureDirectoryExists(image_file_directory)

        graph = self.GenerateGraphRandom()
        pos = nx.planar_layout(graph, scale=0.5, center=(0.5, 0.5))
        self.RenderGraph(graph, pos, f'{image_file_directory}/{base_name}.png')
        self.RecordGraphEdges(graph, pos, f'{label_file_directory}/{base_name}.txt')

    def GenerateDataSets(self):
        EnsureDirectoryExists(self.output_directory)
        # generate the training data set
        for n in range(self.training_samples):
            self.GenerateSample(f'graph_{n}', self.output_directory, True)
        # generate the 
        for n in range(self.validation_samples):
            self.GenerateSample(f'graph_{n}', self.output_directory, False)

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
    if len(sys.argv) < 2:
        print('Destination directory required')
        exit(-1)

    ydg = YoloDataGenerator(sys.argv[1], 20000, 500)
    ydg.GenerateDataSets()
    print('Done!')


# this training command is essential 
# training command
# yolo detect train data=./yolo-training-data/data.yaml model=yolo11s.pt augment=False epochs=16 pretrained=False hsv_h=0.0 hsv_s=0.0 hsv_v=0.0 degrees=0.0 translate=0.0 scale=0.0 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.0 bgr=0.0 mosaic=0.0 mixup=0.0 cutmix=0.0 copy_paste=0.0 erasing=0.0

# we can not have any augmentation because
# 1. only straight edges exist
# 2. augmentation will change and mess up the edge class since the edge class is tied to geometry.
