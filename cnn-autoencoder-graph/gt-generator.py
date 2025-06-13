#!/usr/bin/env python3

import cv2
import numpy as np
import random
import sys

if len(sys.argv) < 2:
    print("supply a directory path")
    exit(0)

ImageSize = 32
MinLineDim = 1
NumSamples = 1000

OutputDirectory = sys.argv[1]

txt_file = open(f'{OutputDirectory}/graphs.txt', 'w+')

for i in range(NumSamples):
    x_1 = x_2 = y_1 = y_2 = 0
    while abs(x_1 - x_2) < MinLineDim and abs(y_1 - y_2) < MinLineDim:
        x_1 = random.randint(0, ImageSize - 1)
        x_2 = random.randint(0, ImageSize - 1)
        y_1 = random.randint(0, ImageSize - 1)
        y_2 = random.randint(0, ImageSize - 1)

    txt_file.write(f'{i}\t{x_1}\t{y_1}\t{x_2}\t{y_2}\n')

    image = np.zeros((ImageSize, ImageSize, 3), dtype=np.uint8)  # Example: black image

    # Define start and end points of the line
    start_point = (x_1, y_1)
    end_point = (x_2, y_2)

    # Define line color (BGR format)
    color = (255, 255, 255) # Blue

    # Define line thickness
    thickness = 1

    # Draw the line
    cv2.line(image, start_point, end_point, color, thickness)
    cv2.imwrite(f'{OutputDirectory}/line_{i}.png', image)


