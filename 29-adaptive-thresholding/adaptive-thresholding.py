import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def adaptiveThresholding():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla-car-2.png')
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    hist = cv.calcHist([imgGray], [0], None, [256], [0, 256])

    plt.figure()
    # plt.subplot(212)
    # plt.plot(hist)
    # plt.title('Histogram')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Number of pixels')

    plt.subplot(221)
    plt.imshow(imgGray, cmap='gray')

    plt.subplot(222)
    _, imgThres = cv.threshold(imgGray, 192, 255, cv.THRESH_BINARY)
    plt.imshow(imgThres, cmap='gray')
    plt.title('Global Thresholding')

    plt.subplot(223)
    maxValue = 255
    blockSize = 7
    offsetC = 2
    imgMean = cv.adaptiveThreshold(imgGray, maxValue, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blockSize, offsetC)
    plt.imshow(imgMean, cmap='gray')
    plt.title('Mean Thresholding')

    plt.subplot(224)
    imgGaus = cv.adaptiveThreshold(imgGray, maxValue, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, blockSize, offsetC)
    plt.imshow(imgGaus, cmap='gray')
    plt.title('Gaussian Thresholding')

    plt.show()

if __name__ == '__main__':
    adaptiveThresholding()