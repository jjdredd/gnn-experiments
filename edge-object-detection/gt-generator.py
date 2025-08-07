#!/usr/bin/env python3

import cv2
import numpy as np
import random
import sys

if len(sys.argv) < 2:
    print("supply a directory path")
    exit(0)

ImageSize = 64
MinLineDim = 3
NumTrainSamples = 80000
NumValSamples = 1000


class YoloDataGenerator():
    def __init__(self, output_directory, training_samples=80000, validation_samples=1000, min_line_dim=5):
        self.output_directory = output_directory
        self.training_samples = training_samples
        self.validation_samples = validation_samples
        self.min_line_dim = min_line_dim
        self.image_size = 640

    @staticmethod
    def PixelCompare(x, y):
        return abs(x - y) < 1.0

    @staticmethod
    def CalculateCenter(p_1, p_2):
        return ((p_1[0] + p_2[0]) / 2.0, (p_1[1] + p_2[1]) / 2.0)

    @staticmethod
    def EdgeBoungdingBox(p_1, p_2):
        return (p_2[0] - p_1[0], p_2[1] - p_1[1])

    def PointInLimits(self, x, y) -> bool:
        return (x > 1 and x < self.image_size - 2 and y > 1 and y < self.image_size - 2)

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

    def GenerateSample(self, base_name, directory_path, training):
        subdirectory = 'train' if training else 'val'
        label_file_directory = f'{directory_path}/labels/{subdirectory}'
        image_file_directory = f'{directory_path}/images/{subdirectory}'
        txt_file = open(f'{label_file_directory}/{base_name}.txt', 'w')
        x_1 = x_2 = y_1 = y_2 = 0
        while (abs(x_1 - x_2) < self.min_line_dim
               and abs(y_1 - y_2) < self.min_line_dim
               and not (self.PointInLimits(x_1, y_1) or self.PointInLimits(x_2, y_2))):
            x_1 = random.randint(5, self.image_size - 5)
            x_2 = random.randint(5, self.image_size - 5)
            y_1 = random.randint(5, self.image_size - 5)
            y_2 = random.randint(5, self.image_size - 5)

        edge_class = self.ClassifyEdge((x_1, y_1), (x_2, y_2))
        edge_center = self.NormalizeCoordinates(YoloDataGenerator.CalculateCenter((x_1, y_1), (x_2, y_2)))
        edge_bb = self.NormalizeCoordinates(YoloDataGenerator.EdgeBoungdingBox((x_1, y_1), (x_2, y_2)))
        # need to normalize coordinates
        txt_file.write(f'{edge_class}\t{edge_center[0]}\t{edge_center[1]}\t{edge_bb[0]}\t{edge_bb[1]}\n')

        image = np.zeros((ImageSize, ImageSize, 3), dtype=np.uint8)  # Example: black image
        # Define start and end points of the line
        start_point = (x_1, y_1)
        end_point = (x_2, y_2)
        # Define line color (BGR format)
        color = (255, 255, 255)
        # Define line thickness
        thickness = 1
        # Draw the line
        cv2.line(image, start_point, end_point, color, thickness)
        cv2.imwrite(f'{image_file_directory}/{base_name}.png', image)

    def GenerateDataSets(self, directory_path):
        pass


