import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def gaussianKernel(size, sigma):
    kernel = cv.getGaussianKernel(size, sigma) # 1D kernel
    kernel = np.outer(kernel, kernel.transpose()) # 2D kernel
    return kernel

def bilateralFilter():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla-car.jpg')
    img = cv.imread(imgPath)   
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height, width, _ = imgRGB.shape
    scale = 1
    width = int(width*scale)
    height = int(height*scale)
    imgRGB = cv.resize(imgRGB, (width, height), interpolation=cv.INTER_LANCZOS4)

    imgFilter = cv.bilateralFilter(imgRGB, d=25, sigmaColor=100, sigmaSpace=100)

    plt.figure()
    plt.subplot(121)
    plt.imshow(imgRGB)

    plt.subplot(122)
    plt.imshow(imgFilter)

    plt.show()

if __name__ == '__main__':
    bilateralFilter()