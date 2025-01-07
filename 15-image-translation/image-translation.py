import cv2 as cv 
import numpy as np
import os

def translation():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imgPath)

    T = np.array([[1, 0, -100],
                  [0, 1, 100]], dtype=np.float32)
    
    height, width = img.shape[:2]

    imgTrans = cv.warpAffine(img, T, (width, height))
    cv.imshow("imgTrans", imgTrans)
    cv.waitKey(0)

if __name__ == '__main__':
    translation()