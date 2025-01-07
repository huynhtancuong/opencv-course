import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def convolution2d():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imgPath)   
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    n = 10
    kernel = np.ones((n,n), np.float32)/(n*n)
    imgFilter = cv.filter2D(imgRGB, ddepth=-1, kernel=kernel)

    plt.figure()
    plt.subplot(121)
    plt.imshow(imgRGB)
    plt.title('Original')

    plt.subplot(122)
    plt.imshow(imgFilter)
    plt.title('Filtered')

    plt.show()


if __name__ == '__main__':
    convolution2d()