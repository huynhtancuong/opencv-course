import cv2 as cv 
import matplotlib.pyplot as plt
import numpy as np
import os 

def histogramBackprojection():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla-car-2.png')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    imgRegion = imgRGB[225:350, 190:350]

    plt.figure()
    plt.subplot(231)
    plt.imshow(imgRGB) 
    plt.title('Original')

    plt.subplot(232)
    plt.imshow(imgRegion)
    plt.title('Region of Interest')

    imgRegionHSV = cv.cvtColor(imgRegion, cv.COLOR_RGB2HSV)
    imgRegionHist = cv.calcHist([imgRegionHSV], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv.normalize(imgRegionHist, imgRegionHist, 0, 255, cv.NORM_MINMAX)
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    out = cv.calcBackProject([imgHSV], [0,1], imgRegionHist, [0, 180, 0, 255], 1)

    plt.subplot(233)
    plt.imshow(out, cmap='hot')
    plt.title('Hist Backprojection')

    ellipseKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    # print(ellipseKernel)
    crossKernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
    
    cv.filter2D(out, -1, ellipseKernel, out)
    plt.subplot(234)
    plt.imshow(out, cmap='hot')    
    plt.title('After Ellipse Kernel')

    _, mask = cv.threshold(out, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    plt.subplot(235)
    plt.imshow(mask, cmap='gray')
    plt.title('Thresholded')

    maskAllChannels = cv.merge((mask, mask, mask))
    result = cv.bitwise_and(imgRGB, maskAllChannels)
    plt.subplot(236)
    plt.imshow(result)
    plt.title('Result')


    plt.show()

if __name__ == '__main__':
    histogramBackprojection()