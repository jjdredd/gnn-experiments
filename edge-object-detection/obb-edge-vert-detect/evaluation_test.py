#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import random
import yaml
import argparse

import matplotlib.pyplot as plt
import random

from ultralytics import YOLO

AnnotationLineThickness = 1


class PredictionRenderer():
    def __init__(self, dest_dir, confidence, model_file=None):
        # Load a model
        self.model = None
        if model_file is not None:
            self.model = YOLO(model_file)  # pretrained YOLO11 model
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

    def RenderLine(self, image, start, end):
        p_1 = start
        p_2 = end
        if not self.model:      # yolo provides image coordinates for some reason
            p_1 *= self.image_size
            p_2 *= self.image_size
        cv2.line(image, tuple(p_1.astype(int)), tuple(p_2.astype(int)),
                 self.dash_color, self.line_thickness)

    def RenderBoundingBox(self, image, center, bounding_box):
        """Here the bounding box is relative to the image size"""
        vec_bbox_half = bounding_box / 2
        vec_width = np.copy(bounding_box)
        vec_width[1] = 0
        self.RenderDashedLine(image, center - vec_bbox_half, center - vec_bbox_half + vec_width)
        self.RenderDashedLine(image, center - vec_bbox_half + vec_width, center + vec_bbox_half)
        self.RenderDashedLine(image, center + vec_bbox_half, center + vec_bbox_half - vec_width)
        self.RenderDashedLine(image, center + vec_bbox_half - vec_width, center - vec_bbox_half)

    def RenderOrientedBoundingBox(self, image, center, bbox, render_cross):
        pass

    def LoadImage(self, image_path):
        image = cv2.imread(image_path)
        assert image is not None
        self.image_size = image.shape[0]
        return image

    def RenderImageInference(self, image_path):
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
                self.RenderOrientedBoundingBox(image, center, bbox)
        cv2.imwrite(os.path.join(self.dest_dir, os.path.basename(image_path)), image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render GT or prediction into an image sample')
    parser.add_argument('--image-path', 
                       help='Path to the image file',
                       required=True)
    parser.add_argument('--model-path', 
                       help='Path to the yolo model weights file',
                       required=True)
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

    if not args.model_path:
        print('Error: either model path or data path must be given')
        exit(-1)
    
    renderer = PredictionRenderer(args.dest_dir, args.confidence, args.model_path)
    renderer.RenderImageInference(args.image_path)
