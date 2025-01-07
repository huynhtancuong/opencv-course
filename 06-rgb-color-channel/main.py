import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt


def pureColors():
    zeros = np.zeros((100,100))
    ones = np.ones((100,100))

    bImg = cv.merge((zeros, zeros, 255*ones))
    gImg = cv.merge((zeros, 255*ones, zeros))
    rImg = cv.merge((255*ones, zeros, zeros))

    black = cv.merge((zeros, zeros, zeros))
    white = cv.merge((255*ones, 255*ones, 255*ones))
    gray = cv.merge((0.5*ones, 0.5*ones, 0.5*ones))

    plt.figure()
    plt.subplot(231)
    plt.imshow(bImg)
    plt.title('Blue')

    plt.subplot(232)
    plt.imshow(gImg)
    plt.title('Green')

    plt.subplot(233)
    plt.imshow(rImg)
    plt.title('Red')

    plt.subplot(234)
    plt.imshow(black)
    plt.title('Black')

    plt.subplot(235)
    plt.imshow(white)
    plt.title('White')

    plt.subplot(236)
    plt.imshow(gray)
    plt.title('Gray')

    plt.show()

def bgrChannelGrayScale():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    b,g,r = cv.split(img)
    
    plt.figure()
    plt.subplot(131)
    plt.imshow(b, cmap='gray')
    plt.title('Blue')

    plt.subplot(132)
    plt.imshow(g, cmap='gray')
    plt.title('Green')

    plt.subplot(133)
    plt.imshow(r, cmap='gray')
    plt.title('Red')

    plt.show()

def bgrChannelColor():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    b,g,r = cv.split(img)

    zeros = np.zeros_like(b)

    bImg = cv.merge((zeros, zeros, b))
    gImg = cv.merge((zeros, g, zeros))
    rImg = cv.merge((r, zeros, zeros))
    
    plt.figure()
    plt.subplot(131)
    plt.imshow(bImg, cmap='gray')
    plt.title('Blue')

    plt.subplot(132)
    plt.imshow(gImg, cmap='gray')
    plt.title('Green')

    plt.subplot(133)
    plt.imshow(rImg, cmap='gray')
    plt.title('Red')

    plt.show()

if __name__ == '__main__':
    # pureColors()
    # bgrChannelGrayScale()
    bgrChannelColor()