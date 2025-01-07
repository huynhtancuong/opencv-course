import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def otsuBinarization():
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

    plt.subplot(131)
    plt.imshow(imgGray, cmap='gray')

    plt.subplot(132)
    thres = 140
    maxVal = 255
    _, imgThres = cv.threshold(imgGray, thres, maxVal, cv.THRESH_BINARY)
    plt.imshow(imgThres, cmap='gray')
    plt.title('Global Thresholding')

    plt.subplot(133)
    arbThres = 0 # we are not using it, because we are using otsu
    _, imgOtsu = cv.threshold(imgGray, arbThres, maxVal, cv.THRESH_BINARY+cv.THRESH_OTSU)
    plt.imshow(imgOtsu, cmap='gray')
    plt.title('Otsu Binarization')

    plt.show()

if __name__ == '__main__':
    otsuBinarization()