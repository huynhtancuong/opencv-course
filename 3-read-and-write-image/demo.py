import cv2 as cv 
import numpy as np
import os 
import matplotlib.pyplot as plt

# Read image
def read_image():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    cv.imshow('Cute', img)
    cv.waitKey(0)

# Write image
def write_image():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    outputPath = os.path.join(root, 'images', 'cute_copy.png')
    cv.imwrite(outputPath, img)

if __name__ == "__main__":
    # read_image()
    write_image()