import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def grayHistogram():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.title('Gray Image')

    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of pixels')

    plt.show()

def colorHistogram():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    imgRGB = cv.imread(imgPath)
    imgRGB = cv.cvtColor(imgRGB, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(imgRGB)
    plt.title('Color Image')

    plt.figure()
    colors = ['b', 'g', 'r']
    for i in range(len(colors)):
        hist = cv.calcHist([imgRGB], [i], None, [256], [0, 256])
        plt.plot(hist, colors[i])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of pixels')

    plt.show()
    
def histogramRegion():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    imgRGB = cv.imread(imgPath)
    imgRGB = cv.cvtColor(imgRGB, cv.COLOR_BGR2RGB)
    imgRGB = imgRGB[50:200, 250:450]

    plt.figure()
    plt.imshow(imgRGB)
    plt.title('Color Image')

    plt.figure()
    colors = ['b', 'g', 'r']
    for i in range(len(colors)):
        hist = cv.calcHist([imgRGB], [i], None, [256], [0, 256])
        plt.plot(hist, colors[i])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of pixels')

    plt.show()

if __name__ == '__main__':
    # grayHistogram()
    # colorHistogram()
    histogramRegion()