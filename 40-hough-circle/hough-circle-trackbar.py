import os 
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt
import time

def houghCircle(img, dp, minDist, param1, param2, minRadius, maxRadius):
    img = img.copy()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    imgBlur = cv.medianBlur(imgGray, 5)
    circles = cv.HoughCircles(imgBlur, cv.HOUGH_GRADIENT, 
                            dp=dp, minDist=minDist, param1=param1,
                            param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    # dp is the inverse ratio of the accumulator resolution to the image resolution
    # minDist is the minimum distance between the centers of the detected circles
    # param1 is the higher threshold for the Canny edge detector
    # param2 is the accumulator threshold for the circle centers
    # minRadius is the minimum circle radius
    # maxRadius is the maximum circle radius
    # circles is a numpy array of (x, y, r) values. 
    # x and y are the coordinates of the center of the circle and r is the radius of the circle

    if circles is None:
        print('No circles detected')
        return imgRGB
    
    circles = np.uint16(np.around(circles))
    # the shape of circles is (1, n, 3) where n is the number of circles detected
    # the 3rd dimension has the values (x, y, r) for each circle
    # so we convert the array to (n, 3) shape
    for circle in circles[0,:]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv.circle(imgRGB, center, radius, (255,0,0), 5)

    return imgRGB

if __name__ == '__main__':
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'coins.jpg')
    img = cv.imread(imgPath)
    # scale down the image
    img = cv.resize(img, (0,0), fx=0.5, fy=0.5)
    winName = 'Hough Circle'
    cv.namedWindow(winName)
    cv.createTrackbar('dp', winName, 1, 10, lambda x: x)
    cv.createTrackbar('minDist', winName, 600, 1000, lambda x: x)
    cv.createTrackbar('param1', winName, 200, 500, lambda x: x)
    cv.createTrackbar('param2', winName, 15, 100, lambda x: x)
    cv.createTrackbar('minRadius', winName, 100, 200, lambda x: x)
    cv.createTrackbar('maxRadius', winName, 150, 300, lambda x: x)

    while True:
        dp = cv.getTrackbarPos('dp', winName)
        minDist = cv.getTrackbarPos('minDist', winName)
        param1 = cv.getTrackbarPos('param1', winName)
        param2 = cv.getTrackbarPos('param2', winName)
        minRadius = cv.getTrackbarPos('minRadius', winName)
        maxRadius = cv.getTrackbarPos('maxRadius', winName)

        # saturate the values
        dp = max(1, dp)
        minDist = max(1, minDist)
        param1 = max(1, param1)
        param2 = max(1, param2)
        minRadius = max(1, minRadius)
        maxRadius = max(1, maxRadius)

        imgCircle = houghCircle(img, dp, minDist, param1, param2, minRadius, maxRadius)
        imgCircle_BGR = cv.cvtColor(imgCircle, cv.COLOR_RGB2BGR)
        cv.imshow(winName, imgCircle_BGR)

        if cv.waitKey(1000) & 0xFF == ord('q'):
            break
        # delay of 1 second
        # time.sleep(1)