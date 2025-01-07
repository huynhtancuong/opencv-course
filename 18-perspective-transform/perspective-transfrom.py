import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

def perspective_transform():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'perspective.jpg')
    img = cv.imread(imgPath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    height, width, _ = img.shape

    # Define the transformation matrix
    # we need 4 points from the original image and 4 points from the new image
    p1 = np.array([[363, 100],
                   [441, 70],
                   [441, 145],
                   [364, 154]], np.float32) # Original points
    p2 = np.array([[0, 0],
                    [width, 0],
                    [width, height],
                    [0, height]], np.float32) # New points
    
    T = cv.getPerspectiveTransform(p1, p2)

    imgTrans = cv.warpPerspective(img, T, (width, height))

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgTrans)
    plt.show()

if __name__ == '__main__':
    perspective_transform()