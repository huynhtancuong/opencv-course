import cv2 as cv 
import matplotlib.pyplot as plt 
import os 
import numpy as np

def templateMatching():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla-car-2.png')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    teslaLogo = imgRGB[307:319, 675:695]
    height, width = teslaLogo.shape[:2]

    plt.figure()
    plt.subplot(121)
    plt.imshow(imgRGB)

    plt.subplot(122)
    plt.imshow(teslaLogo)

    methods = [cv.TM_CCOEFF,
               cv.TM_CCOEFF_NORMED,
               cv.TM_CCORR,
               cv.TM_CCORR_NORMED,
               cv.TM_SQDIFF,
               cv.TM_SQDIFF_NORMED]
    
    method_names = ['TM_CCOEFF',
                    'TM_CCOEFF_NORMED',
                    'TM_CCORR',
                    'TM_CCORR_NORMED',
                    'TM_SQDIFF',
                    'TM_SQDIFF_NORMED']
    
    for i, method in enumerate(methods):
        curImg = imgRGB.copy()
        templateMap = cv.matchTemplate(img, teslaLogo, method)
        _,_,minLoc,maxLoc = cv.minMaxLoc(templateMap)

        if method == cv.TM_SQDIFF or method == cv.TM_SQDIFF_NORMED:
            topLeft = minLoc
        else:
            topLeft = maxLoc

        bottomRight = (topLeft[0]+width, topLeft[1]+height)
        cv.rectangle(curImg, topLeft, bottomRight, (255,0,0), 5)
        plt.figure()
        plt.subplot(211)
        plt.imshow(templateMap)
        plt.title(method_names[i])
        plt.subplot(212)
        plt.imshow(curImg)

    plt.show()

if __name__ == '__main__':
    templateMatching()