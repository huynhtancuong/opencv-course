import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt
from math import exp

def imageBlending():
    root = os.getcwd()
    imagePath1 = os.path.join(root, 'images', 'cute.jpeg')
    imagePath2 = os.path.join(root, 'images', 'universe.jpg')
    img1 = cv.imread(imagePath1)
    img2 = cv.imread(imagePath2)

    img1 = cv.resize(img1, (img2.shape[1], img2.shape[0]))

    windowName = 'image blending'
    cv.namedWindow(windowName)
    
    scale = 100
    cv.createTrackbar('alpha', windowName, 0, 1*scale, lambda x: None)
    cv.createTrackbar('gamma', windowName, 0, 255, lambda x: None)

    while True:
        if cv.waitKey(1) == ord('q'):
            break
        alpha = cv.getTrackbarPos('alpha', windowName)/scale
        beta = lambda x: 1-x
        gamma = cv.getTrackbarPos('gamma', windowName)

        imgBlend = cv.addWeighted(img1, alpha, img2, beta(alpha), gamma)

        cv.imshow(windowName, imgBlend) 

    cv.destroyAllWindows()

if __name__ == '__main__':
    imageBlending()