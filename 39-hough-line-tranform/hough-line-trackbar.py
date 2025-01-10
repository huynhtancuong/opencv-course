import os 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def houghLineTransform(img, canny_thres: tuple, hough_thres: int):
    imgBlur = cv.GaussianBlur(img, (5,5), 3)
    low_thres, high_thres = canny_thres
    cannyEdges = cv.Canny(imgBlur, low_thres, high_thres)

    distResol = 1 # rho (distance resolution)
    angleResol = np.pi/180 # theta (angle resolution)
    threshold = hough_thres
    lines = cv.HoughLines(cannyEdges, distResol, angleResol, threshold)
    imgLines = img.copy()

    k = 1000 # scale factor, to draw lines

    if lines is None:
        return imgLines

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

    cannyAndLines = np.hstack((cannyEdges, imgLines))

    return cannyAndLines


if __name__ == '__main__':
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla-car-2.png')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    winName = 'Hough Line Transform'
    cv.namedWindow(winName)
    cv.createTrackbar('minThres', winName, 100, 255, lambda x: None)
    cv.createTrackbar('maxThres', winName, 255, 255, lambda x: None)
    cv.createTrackbar('houghThres', winName, 150, 255, lambda x: None)

    while True:
        minThres = cv.getTrackbarPos('minThres', winName)
        maxThres = cv.getTrackbarPos('maxThres', winName)
        houghThres = cv.getTrackbarPos('houghThres', winName)

        imgLines = houghLineTransform(img, (minThres, maxThres), houghThres)
        cv.imshow(winName, imgLines)

        if cv.waitKey(20) == ord('q'):
            break