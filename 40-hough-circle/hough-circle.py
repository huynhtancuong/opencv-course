import os 
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt

def houghCircle():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'coins.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    imgBlur = cv.medianBlur(imgGray, 5)
    circles = cv.HoughCircles(imgBlur, cv.HOUGH_GRADIENT, 
                              dp=1, minDist=600, param1=200,
                              param2=15, minRadius=100, maxRadius=150)
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
        return
    
    circles = np.uint16(np.around(circles))
    # the shape of circles is (1, n, 3) where n is the number of circles detected
    # the 3rd dimension has the values (x, y, r) for each circle
    # so we convert the array to (n, 3) shape
    for circle in circles[0,:]:
        center = (circle[0], circle[1])
        radius = circle[2]
        cv.circle(imgRGB, center, radius, (255,0,0), 5)

    plt.figure()
    plt.imshow(imgRGB)
    plt.show()

if __name__ == '__main__':
    houghCircle()