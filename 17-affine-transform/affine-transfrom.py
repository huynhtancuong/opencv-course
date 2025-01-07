import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def affine_transform():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    height, width, _ = img.shape

    # Define the transformation matrix
    # we need 3 points from the original image and 3 points from the new image
    p1 = np.array([[50, 50], [200, 50], [50, 200]], np.float32) # Original points
    p2 = np.array([[10, 100], [200, 50], [100, 250]], np.float32) # New points

    T = cv.getAffineTransform(p1, p2)
    imgTrans = cv.warpAffine(img, T, (width, height))

    print(T)

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgTrans)
    plt.show()

if __name__ == '__main__':
    affine_transform()