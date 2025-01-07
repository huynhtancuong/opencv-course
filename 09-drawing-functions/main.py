import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt


def drawingFunction():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # plt.figure()
    # plt.imshow(imgRGB)
    # plt.show()

    white = (255, 255, 255)
    black = (0, 0, 0)
    green = (0, 255, 0)
    cv.line(img, (100, 100), (200, 200), color=white, thickness=5)

    r,c,d = img.shape
    offset = 50
    cv.rectangle(img, pt1=(offset, offset), pt2=(c-offset, r-offset), color=white, thickness=5)

    cv.circle(img, center=(c//2, r//2), radius=50, color=white, thickness=-1)

    cv.ellipse(img, center=(c//2, r//2), axes=(100, 200), angle=45, startAngle=0, endAngle=180, color=white, thickness=-1)

    pts = np.array([[10,5], [20,30], [70,20], [50,10]], np.int32)

    cv.polylines(img, pts=[pts], isClosed=True, color=white, thickness=5)

    cv.putText(img, text='OpenCV', org=(10, 200), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=4, color=green, thickness=5)

    cv.imshow('Line', img)
    cv.waitKey(0)


if __name__ == '__main__':
    drawingFunction()