import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def cannyEdgeDetection():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla-car-2.png')
    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height, width = img.shape[:2]
    scale = 1/2
    height, width = int(height*scale), int(width*scale)
    img = cv.resize(img, (width, height), interpolation=cv.INTER_LINEAR)

    winName = 'Canny Edge Detection'
    cv.namedWindow(winName)
    cv.createTrackbar('minThres', winName, 0, 255, lambda x: None)
    cv.createTrackbar('maxThres', winName, 0, 255, lambda x: None)

    while True:
        minThres = cv.getTrackbarPos('minThres', winName)
        maxThres = cv.getTrackbarPos('maxThres', winName)

        edges = cv.Canny(img, minThres, maxThres)
        cv.imshow(winName, edges)

        if cv.waitKey(20) == ord('q'):
            break

if __name__ == '__main__':
    cannyEdgeDetection()