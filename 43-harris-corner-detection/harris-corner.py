import cv2 as cv 
import os 
import numpy as np
import matplotlib.pyplot as plt

def harrisCorner():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'chessboard.png')
    img = cv.imread(imgPath)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(imgGray, cmap='gray')
    plt.title('Gray Image')

    # Harris corner detection
    blockSize = 5
    sobelSize = 3
    k = 0.04
    harris = cv.cornerHarris(imgGray, blockSize, sobelSize, k)
    plt.subplot(1, 3, 2)
    plt.imshow(harris, cmap='hot')
    plt.title('Harris Corner')

    # Thresholding the harris corner
    imgRGB[harris > 0.01*harris.max()] = [255, 0, 0]
    plt.subplot(1, 3, 3)
    plt.imshow(imgRGB)
    plt.title('Harris Corner')

    plt.show()

if __name__ == '__main__':
    harrisCorner()