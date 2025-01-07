import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def morphTrans():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla-car-2.png')
    imgGray = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    maxValue = 255
    blockSize = 7
    offsetC = 3

    plt.figure()
    plt.subplot(241)
    imgGaus = cv.adaptiveThreshold(imgGray, maxValue, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, blockSize, offsetC)
    imgGaus = cv.GaussianBlur(imgGaus, (7,7), sigmaX=2)
    plt.imshow(imgGaus, cmap='gray')
    plt.title('Gaus Thres')

    plt.subplot(242)
    kernel = np.ones((7,7), np.uint8)
    erosion = cv.erode(imgGaus, kernel, iterations=1)
    plt.imshow(erosion, cmap='gray')
    plt.title('Erosion')

    plt.subplot(243)
    dilation = cv.dilate(imgGaus, kernel, iterations=1)
    plt.imshow(dilation, cmap='gray')
    plt.title('Dilation')

    morphTypes = [cv.MORPH_OPEN,
                  cv.MORPH_CLOSE,
                  cv.MORPH_GRADIENT,
                  cv.MORPH_TOPHAT,
                  cv.MORPH_BLACKHAT]
    morphNames = ['Open', 'Close', 'Gradient', 'TopHat', 'BlackHat']

    for i in range(len(morphTypes)):
        plt.subplot(2, 4, i+4)
        morph = cv.morphologyEx(imgGaus, morphTypes[i], kernel)
        plt.imshow(morph, cmap='gray')
        plt.title(morphNames[i])

    plt.show()

    ellipseKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    print(ellipseKernel)
    crossKernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
    print(crossKernel)

if __name__ == '__main__':
    morphTrans()