import os 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

"""
OpenCV has the function, cv.grabCut() for this. We will see its arguments first:
    img - Input image
    mask - It is a mask image where we specify which areas are background, foreground or probable background/foreground etc. It is done by the following flags, cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD, or simply pass 0,1,2,3 to image.
    rect - It is the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
    bdgModel, fgdModel - These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays of size (1,65).
    iterCount - Number of iterations the algorithm should run.
    mode - It should be cv.GC_INIT_WITH_RECT or cv.GC_INIT_WITH_MASK or combined which decides whether we are drawing rectangle or final touchup strokes.
"""

def graphCutSegmentation():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cr7.jpg')
    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    assert img is not None, "File not found"
    
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    rect = (240, 10, 888, 551)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=1, mode=cv.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img = img*mask2[:,:, np.newaxis]

    plt.figure()
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    graphCutSegmentation()