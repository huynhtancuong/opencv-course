import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def thresholding():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla.jpg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    hist = cv.calcHist([img], [0], None, [256], [0, 256])

    plt.figure()
    plt.plot(hist)
    plt.title('Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of pixels')

    plt.show()

    thresOpt = [cv.THRESH_BINARY,
                cv.THRESH_BINARY_INV,
                cv.THRESH_TRUNC,
                cv.THRESH_TOZERO,
                cv.THRESH_TOZERO_INV]
    
    thresNames = ['Binary', 'Binary Inv', 'Trunc', 'Tozero', 'Tozero Inv']

    thresValue = 143

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')

    for i in range(len(thresOpt)):
        plt.subplot(2, 3, i+2)
        ret, imgThres = cv.threshold(img, thresValue, 255, thresOpt[i])
        plt.imshow(imgThres, cmap='gray')
        plt.title(thresNames[i])

    plt.show()

if __name__ == '__main__':
    thresholding()