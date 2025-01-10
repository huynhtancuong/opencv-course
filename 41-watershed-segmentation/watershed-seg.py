import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np
import os 


def watershedSegmentation():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'water_coins.jpg')
    img = cv.imread(imgPath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    assert img is not None, 'Image not found'
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgGray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    plot_rows = 2
    plot_cols = 3

    plt.figure()
    plt.subplot(plot_rows, plot_cols, 1)
    plt.imshow(thresh, cmap='gray')
    plt.title('Thresholded Image')

    # Remove any noise in the image
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    plt.subplot(plot_rows, plot_cols, 2)
    plt.imshow(opening, cmap='gray')
    plt.title('Noise Removal')

    # Sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    plt.subplot(plot_rows, plot_cols, 3)
    plt.imshow(sure_bg, cmap='gray')
    plt.title('Sure Background')

    # Finding sure foreground area
    # Because the coins are touching each other, we can't use erosion to find the sure foreground
    # Instead, we use distance transform
    dist_transform = cv.distanceTransform(opening, distanceType=cv.DIST_L2, maskSize=5)
    plt.subplot(plot_rows, plot_cols, 4)
    plt.imshow(dist_transform, cmap='gray')
    plt.title('Distance Transform')

    ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    plt.subplot(plot_rows, plot_cols, 5)
    plt.imshow(sure_fg, cmap='gray')
    plt.title('Sure Foreground')

    # Finding unknown region
    # Subtract sure foreground from sure background
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    plt.subplot(plot_rows, plot_cols, 6)
    plt.imshow(unknown, cmap='gray')
    plt.title('Unknown Region')

    # Marker labelling
    # Now we know for sure which are region of coins, which are background and all. 
    # So we create marker (it is an array of same size as that of original image, but 
    # with int32 datatype) and label the regions inside it. The regions we know for sure 
    # (whether foreground or background) are labelled with any positive integers, 
    # but different integers, and the area we don't know for sure are just left as zero. 
    # For this we use cv.connectedComponents(). It labels background of the image with 0, 
    # then other objects are labelled with integers starting from 1.

    ret, labels = cv.connectedComponents(sure_fg)

    # cv.watershed() function considers the regions with value 0 as unknown regions
    # so we change the value of unknown regions in variable labels to 1
    labels = labels + 1

    # Now, mark the region of unknown with 0
    labels[unknown == 255] = 0

    # Apply watershed algorithm
    labels = cv.watershed(imgRGB, labels)
    # Enhance the boundaries
    imgRGB[labels == -1] = [255, 0, 0]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(labels, cmap='flag')
    plt.title('Labels')
    plt.subplot(1, 2, 2)
    plt.imshow(imgRGB)
    plt.title('Segmented Image')

    
    plt.show()



if __name__ == '__main__':
    watershedSegmentation()