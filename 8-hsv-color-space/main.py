import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt

def hsvColorSegmentation():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    imgHSV = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    lower = np.array([89, 149, 0])
    upper = np.array([112, 255, 255])

    mask = cv.inRange(imgHSV, lower, upper)
    mask = cv.bitwise_not(mask)
    
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.show()


if __name__ == '__main__':
    hsvColorSegmentation()
