import cv2 as cv 
import numpy as np 
import os 
import matplotlib.pyplot as plt

def pyramidBlending():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    imgBGR = cv.imread(imgPath)

    # make sure the size of the image is a power of 2.
    # if not, resize it to the nearest power of 2.
    # this is necessary for the pyramid blending to work.
    h, w = imgBGR.shape[:2]
    h = 2 ** int(np.ceil(np.log2(h)))
    w = 2 ** int(np.ceil(np.log2(w)))
    imgBGR = cv.resize(imgBGR, (w, h), interpolation=cv.INTER_LINEAR)

    # Process the first image
    plt.figure()
    downSamp = imgBGR.copy()
    BGR_gaussian_pyramid = [downSamp]
    plt.subplot(231)
    plt.imshow(downSamp)
    for i in range(5):
        plt.subplot(2, 3, i+2)
        downSamp = cv.pyrDown(downSamp)
        plt.imshow(downSamp)
        BGR_gaussian_pyramid.append(downSamp)

    plt.figure()
    BGR_laplacian_pyramid = [BGR_gaussian_pyramid[4]]
    for i in range(4, 0, -1):
        plt.subplot(2, 3, 4-i+1)
        upSamp = cv.pyrUp(BGR_gaussian_pyramid[i])
        laplacian = cv.subtract(BGR_gaussian_pyramid[i-1], upSamp)
        BGR_laplacian_pyramid.append(laplacian)
        plt.imshow(laplacian)

    # Process the second image
    imgRGB = cv.cvtColor(imgBGR, cv.COLOR_BGR2RGB)
    downSamp = imgRGB.copy()
    RGB_gaussian_pyramid = [downSamp]
    plt.figure()
    plt.subplot(231)
    plt.imshow(downSamp)
    for i in range(5):
        plt.subplot(2, 3, i+2)
        downSamp = cv.pyrDown(downSamp)
        plt.imshow(downSamp)
        RGB_gaussian_pyramid.append(downSamp)

    plt.figure()
    RGB_laplacian_pyramid = [RGB_gaussian_pyramid[4]]
    for i in range(4, 0, -1):
        plt.subplot(2, 3, 4-i+1)
        upSamp = cv.pyrUp(RGB_gaussian_pyramid[i])
        laplacian = cv.subtract(RGB_gaussian_pyramid[i-1], upSamp)
        RGB_laplacian_pyramid.append(laplacian)
        plt.imshow(laplacian)

    

    # Blend the two images
    plt.figure()
    combined_list = []
    offset = 3
    for i in range(len(BGR_laplacian_pyramid)):
        left = RGB_laplacian_pyramid[i]
        right = BGR_laplacian_pyramid[i]
        _, cols = left.shape[:2]
        combined = np.hstack((left[:, :cols//2 + offset], right[:, cols//2 + offset:]))
        combined_list.append(combined)
        plt.subplot(2, 3, i+1)
        plt.imshow(combined)

    blend = combined_list[0]
    for i in range(1, 4):
        blend = cv.pyrUp(blend)
        blend = cv.add(blend, combined_list[i])

    plt.figure()
    plt.imshow(blend)
    
    plt.show()




if __name__ == '__main__':
    pyramidBlending()