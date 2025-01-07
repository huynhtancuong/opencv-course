import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt


def readAndWritePixel():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

    eyePixel = imgRGB[150, 364]
    imgRGB[150, 364] = [255, 0, 0]
    
    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

def readAndWritePixelRegion():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

    eyeRegion = imgRGB[105:135, 300:330]

    plt.figure()
    plt.imshow(eyeRegion)
    plt.show()

    startX, startY = 338, 110
    dx, dy = eyeRegion.shape[1], eyeRegion.shape[0]

    imgRGB[startY:startY+dy, startX:startX+dx] = eyeRegion

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

if __name__ == '__main__':
    # readAndWritePixel()
    readAndWritePixelRegion()