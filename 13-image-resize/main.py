import cv2 as cv 
import os 
import numpy as np
import matplotlib.pyplot as plt

def resize_image():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img[110:135, 300:330, :]
    height, width = img.shape[:2]

    scale = 1/2

    interpMethods = [cv.INTER_AREA,
                     cv.INTER_LINEAR,
                     cv.INTER_NEAREST,
                     cv.INTER_CUBIC,
                     cv.INTER_LANCZOS4]

    interpTitles = ['INTER_AREA',
                    'INTER_LINEAR',
                    'INTER_NEAREST',
                    'INTER_CUBIC',
                    'INTER_LANCZOS4']
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    
    for i, method in enumerate(interpMethods):
        plt.subplot(2, 3, i+2)
        imgResize = cv.resize(img, (int(width*scale), int(height*scale)),
                              interpolation=method)
        plt.imshow(imgResize)
        plt.title(interpTitles[i])

    plt.show()



if __name__ == "__main__":
    resize_image()