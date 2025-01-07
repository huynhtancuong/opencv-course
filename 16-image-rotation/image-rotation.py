import cv2 as cv 
import numpy as np
import os 

def rotation():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imgPath)
    height, width, _ = img.shape

    # Rotation matrix
    T = cv.getRotationMatrix2D(center=(width/2, height/2), angle=45, scale=1)
    rotated_img = cv.warpAffine(img, T, (width, height))

    cv.imshow('Rotated Image', rotated_img)
    cv.waitKey(0)

if __name__ == "__main__":
    rotation()