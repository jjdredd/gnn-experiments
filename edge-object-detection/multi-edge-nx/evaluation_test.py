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

import numpy as np

from ultralytics import YOLO

ImageSize = 400
AnnotationLineThickness = 1


class Inference():
    def __init__(self, model_file):
        # Load a model
        self.model = YOLO(model_file)  # pretrained YOLO11n model
        self.image_size = ImageSize
        self.dash_length = 2 / self.image_size
        self.dash_interval = 3 / self.image_size
        self.line_color = (255, 255, 255)
        self.line_thickness = AnnotationLineThickness
        self.eps = 10**(-3)

    def Inference(self, image_directory):
        # Run batched inference on a list of images
        image_list = []
        for file_name in os.listdir(image_directory):
            if not os.path.isfile(file_name):
                continue
            print(file_name)
            image_list.append(file_name)
        return self.model(image_list)

    def FuzzyCompare(self, x, y):
        return abs(x - y) < self.eps

    def FuzzyCompare2D(self, x, y):
        return self.FuzzyCompare(x[0], y[0]) and self.FuzzyCompare(x[1], y[1])

    @staticmethod
    def PointInLine(start, end, point):
        v_1 = end - start
        v_2 = point - end
        return np.dot(v_1, v_2) > 0

    def RenderDashedLine(self, image, start, end):
        direction = end - start
        direction /= (np.linalg.norm(direction) + self.eps)
        p_current = start
        p_next = p_current + self.dash_length * direction
        while Inference.PointInLine(start, end, p_next):
            # draw the line
            p_current = p_next + self.dash_interval * direction
            p_next = p_current + self.dash_length * direction

    def RenderLine(self, image, start, end):
        cv2.line(image, start, end, self.line_color, self.line_thickness)

    def RenderBoundingBox(self, image, center, bounding_box):
        pass

    def RenderResult(self, image, edge_class, center, bounding_box):
        pass

    def RenderImageResults(self, image, results):
        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            masks = result.masks  # Masks object for segmentation masks outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs
            probs = result.probs  # Probs object for classification outputs
            obb = result.obb  # Oriented boxes object for OBB outputs
            result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk



if __name__ == '__main__':
    model_file = "yolo11n.pt"
    if len(sys.argv) < 2:
        print('Image directory required')
        exit(-1)

    inference = Inference(model_file)
    inference.Inference(sys.argv[1])
    print('Done!')
