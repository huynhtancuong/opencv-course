import os 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def houghLineTransform():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla-car.jpg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    imgBlur = cv.GaussianBlur(img, (5,5), 3)
    cannyEdges = cv.Canny(imgBlur, 100, 200)

    plt.figure()
    plt.subplot(411)
    plt.imshow(img)
    plt.title('Original')

    plt.subplot(412)
    plt.imshow(imgBlur, cmap='gray')
    plt.title('Blurred')

    plt.subplot(413)
    plt.imshow(cannyEdges, cmap='gray')
    plt.title('Canny Edges')

    distResol = 1 # rho (distance resolution)
    angleResol = np.pi/180 # theta (angle resolution)
    threshold = 150
    lines = cv.HoughLines(cannyEdges, distResol, angleResol, threshold)
    imgLines = img.copy()

    k = 1000 # scale factor, to draw lines

    for line in lines:
        rho, theta = line[0] 
        dhat = np.array([np.cos(theta), np.sin(theta)])
        d = rho*dhat
        lhat = np.array([-np.sin(theta), np.cos(theta)])
        p1 = d + k*lhat
        p2 = d - k*lhat
        p1 = p1.astype(int)
        p2 = p2.astype(int)
        cv.line(imgLines, tuple(p1), tuple(p2), (255,255,255), 2)

    plt.subplot(414)
    plt.imshow(imgLines, cmap='gray')
    plt.title('Hough Lines')

    plt.show()


if __name__ == '__main__':
    houghLineTransform()