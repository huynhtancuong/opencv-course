import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def imageGradient():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla-car-2.png')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    plt.figure()
    plt.subplot(221)
    plt.imshow(img, cmap='gray')
    plt.title('Original')

    laplacian = cv.Laplacian(img, cv.CV_64F, ksize=13)
    plt.subplot(222)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian')

    # We can create the sobel kernels manually
    kx, ky = cv.getDerivKernels(1, 0, 3)
    print(ky@kx.T)
    # Or we can use the cv.Sobel function
    sobelX = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=21)
    plt.subplot(223)
    plt.imshow(sobelX, cmap='gray')
    plt.title('Sobel X')

    kx, ky = cv.getDerivKernels(0, 1, 3)
    print(ky@kx.T)
    sobelY = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=21)
    plt.subplot(224)
    plt.imshow(sobelY, cmap='gray')
    plt.title('Sobel Y')



    plt.show()


if __name__ == '__main__':
    imageGradient()