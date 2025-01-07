import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def borderPadding():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'universe.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    pad = 200

    borderTypes = [cv.BORDER_CONSTANT, 
                   cv.BORDER_REPLICATE, 
                   cv.BORDER_REFLECT, 
                   cv.BORDER_WRAP, 
                   cv.BORDER_REFLECT_101]
    
    borderTitles = ['Constant', 
                    'Replicate', 
                    'Reflect', 
                    'Wrap', 
                    'Reflect 101']
    
    plt.figure()
    plt.subplot(231)
    plt.imshow(imgRGB)
    plt.title('Original')
    
    for i, type, title in zip(range(len(borderTypes)), borderTypes, borderTitles):
        imgBorder = cv.copyMakeBorder(imgRGB, pad, pad, pad, pad, type)
        plt.subplot(232+i)
        plt.imshow(imgBorder)
        plt.title(title)

    plt.show()


if __name__ == '__main__':
    borderPadding()