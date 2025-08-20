#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import random
import yaml
import argparse

import networkx as nx
import matplotlib.pyplot as plt
import random

from ultralytics import YOLO

AnnotationLineThickness = 1


class PredictionRenderer():
    def __init__(self, dest_dir, confidence, model_file=None):
        # Load a model
        self.model = None
        if model_file is not None:
            self.model = YOLO(model_file)  # pretrained YOLO11n model
        self.dest_dir = dest_dir
        self.confidence = confidence
        self.image_size = None
        self.dash_length = 0.02
        self.dash_interval = 0.03
        if self.model:          # in pixels
            self.dash_length = 3
            self.dash_interval = 4

        self.dash_color = (0, 0, 255)
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
        v_2 = end - point
        return np.dot(v_1, v_2) > 0

    def RenderLine(self, image, start, end):
        p_1 = start
        p_2 = end
        if not self.model:      # yolo provides image coordinates for some reason
            p_1 *= self.image_size
            p_2 *= self.image_size
        cv2.line(image, tuple(p_1.astype(int)), tuple(p_2.astype(int)),
                 self.dash_color, self.line_thickness)

    def RenderDashedLine(self, image, start, end):
        direction = end - start
        direction /= (np.linalg.norm(direction) + self.eps)
        p_current = start
        p_next = p_current + self.dash_length * direction
        while PredictionRenderer.PointInLine(start, end, p_next):
            self.RenderLine(image, p_current, p_next)
            p_current = p_next + self.dash_interval * direction
            p_next = p_current + self.dash_length * direction

    def RenderBoundingBox(self, image, center, bounding_box):
        """Here the bounding box is relative to the image size"""
        vec_bbox_half = bounding_box / 2
        vec_width = np.copy(bounding_box)
        vec_width[1] = 0
        self.RenderDashedLine(image, center - vec_bbox_half, center - vec_bbox_half + vec_width)
        self.RenderDashedLine(image, center - vec_bbox_half + vec_width, center + vec_bbox_half)
        self.RenderDashedLine(image, center + vec_bbox_half, center + vec_bbox_half - vec_width)
        self.RenderDashedLine(image, center + vec_bbox_half - vec_width, center - vec_bbox_half)

    def RenderBboxEdge(self, image, edge_class, center, bounding_box):
        vec_bbox_half = bounding_box / 2
        if edge_class == 0:
            vec_bbox_half[1] *= -1 # posslant upside-down coordinates
        elif edge_class == 2:
            vec_bbox_half[1] = 0 # horizontal
        elif edge_class == 3:
            vec_bbox_half[0] = 0 # vertical
        self.RenderDashedLine(image, center - vec_bbox_half, center + vec_bbox_half)

    def LoadImage(self, image_path):
        image = cv2.imread(image_path)
        assert image is not None
        self.image_size = image.shape[0]
        return image

    def RenderImageGtData(self, image_path, data_path, render_mode):
        image = self.LoadImage(image_path)
        annotations = np.loadtxt(data_path)
        if annotations.ndim == 1:
            annotations = annotations.reshape(1, -1)
        if render_mode == 'line':
            for a in annotations:
                center = np.array([a[1], a[2]])
                bbox = np.array([a[3], a[4]])
                self.RenderBboxEdge(image, int(a[0]), center, bbox)
        elif render_mode == 'box':
            for a in annotations:
                center = np.array([a[1], a[2]])
                bbox = np.array([a[3], a[4]])
                self.RenderBoundingBox(image, center, bbox)
        cv2.imwrite(os.path.join(self.dest_dir, os.path.basename(image_path)), image)

    def RenderImageInference(self, image_path, render_mode):
        if self.model is None:
            print('Error: Model not loaded and inference is requested')
            return

        image = self.LoadImage(image_path)
        results = self.model.predict(image, save=False, imgsz=self.image_size, conf=self.confidence)
        for result in results:
            yolo_boxes = result.boxes.cpu().numpy()
            for n, cls in enumerate(yolo_boxes.cls):
                bxywh = yolo_boxes.xywh[n]
                center = np.array([bxywh[0], bxywh[1]]) # / self.image_size
                bbox = np.array([bxywh[2], bxywh[3]]) # / self.image_size
                if render_mode == 'line':
                    self.RenderBboxEdge(image, cls, center, bbox)
                elif render_mode == 'box':
                    self.RenderBoundingBox(image, center, bbox)
        cv2.imwrite(os.path.join(self.dest_dir, os.path.basename(image_path)), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render GT or prediction into an image sample')
    parser.add_argument('--image-path', 
                       help='Path to the image file',
                       required=True)
    parser.add_argument('--data-path', 
                       help='Path to the data file',
                       required=False)
    parser.add_argument('--model-path', 
                       help='Path to the yolo model weights file',
                       required=False)
    # parser.add_argument('--image-size',
    #                    type=int,
    #                    default=400,
    #                    help='Size of square image (width and height in pixels)',
    #                    required=False)
    parser.add_argument('--render-mode', 
                       type=str,
                       choices=['line', 'box'],
                       default='line',
                       help='Render mode (line or box, default: line)')
    parser.add_argument('--dest-dir', 
                        help='Destination directory path',
                        required=True)
    parser.add_argument('--confidence', 
                       type=float,
                       default=0.5,
                       help='Confidence threshold (default: 0.5) unused now')
    args = parser.parse_args()
    if args.dest_dir:
        try:
            os.makedirs(args.dest_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {args.dest_dir}: {e}")
            exit(-1)

    if not args.data_path and not args.model_path:
        print('Error: either model path or data path must be given')
        exit(-1)
    
    renderer = PredictionRenderer(args.dest_dir, args.confidence, args.model_path)
    if args.model_path:
        renderer.RenderImageInference(args.image_path, args.render_mode)
    else:
        renderer.RenderImageGtData(args.image_path, args.data_path, args.render_mode)

    # two sources of detections:
    # 1. the ground truth file of yolo format
    # 2. the model prediction (model should be run in this script)

    # two modes of rendering:
    # 1. bounding box (red dashed)
    # 2. edge (red dashed)
