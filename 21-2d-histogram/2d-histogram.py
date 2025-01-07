import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt 
import os 

def histogram2d():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

    plt.figure()
    plt.subplot(131)
    plt.imshow(imgRGB)

    plt.subplot(132)
    plt.imshow(hist, cmap='hot')
    plt.xlabel('Saturation')
    plt.ylabel('Hue')

    lowerBound = np.array([94, 252, 0])
    upperBound = np.array([100, 255, 255])

    mask = cv.inRange(hsv, lowerBound, upperBound)

    plt.subplot(133)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.show()

    plt.figure()
    ax = plt.axes(projection='3d')
    x = np.arange(0, 256, 1)
    y = np.arange(0, 180, 1)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, hist)
    plt.show()


    

if __name__ == '__main__':
    histogram2d()