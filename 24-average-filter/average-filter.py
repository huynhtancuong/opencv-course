import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def averageFilter():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imgPath)   

    windowName = 'Average Filter'
    cv.namedWindow(winname=windowName)
    cv.createTrackbar('n', windowName, 1, 100, lambda x: None)

    while True:
        n = cv.getTrackbarPos('n', windowName)
        if n < 1:
            n = 1
        imgFilter = cv.blur(img, (n,n))
        cv.imshow(windowName, imgFilter)
        
        if cv.waitKey(20) == ord('q'):
            break

    cv.destroyAllWindows()



if __name__ == '__main__':
    averageFilter()