import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt

def trackbar():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)

    windowName = 'Trackbar'
    cv.namedWindow(windowName)
    cv.createTrackbar('B', windowName, 0, 255, lambda x: None)
    cv.createTrackbar('G', windowName, 0, 255, lambda x: None)
    cv.createTrackbar('R', windowName, 0, 255, lambda x: None)

    while True:
        cv.imshow(windowName, img)
        if cv.waitKey(20) == ord('q'):
            cv.destroyAllWindows()
            break

        b = cv.getTrackbarPos('B', windowName)
        g = cv.getTrackbarPos('G', windowName)
        r = cv.getTrackbarPos('R', windowName)

        cv.circle(img, center=(img.shape[1]//2, img.shape[0]//2), radius=50, color=(b,g,r), thickness=-1)
    
    cv.destroyAllWindows()

if __name__ == '__main__':
    trackbar()