import os 
import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

def contour():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'tesla-car-2.png')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    img = img[280:320, 360:400]

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Original')

    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # plt.subplot(232)
    # plt.imshow(thresh, cmap='gray')
    # plt.title('Thresholded')

    kernel = np.ones((3,3), np.uint8)
    dilation = cv.dilate(thresh, kernel, iterations=1)
    plt.subplot(232)
    plt.imshow(dilation, cmap='gray')
    plt.title('Dilated')

    contours, _ = cv.findContours(dilation, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = [contours[0]]
    cv.drawContours(img, contours, -1, (0,0,255), 3)

    plt.subplot(233)
    plt.imshow(img, cmap='gray')
    plt.title('Contour')

    M = cv.moments(contours[0])
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    plt.subplot(234)
    plt.imshow(img, cmap='gray')
    plt.scatter(cx, cy, color='red')
    plt.title('Centroid')

    area = cv.contourArea(contours[0])
    perimeter = cv.arcLength(contours[0], True)

    epsilon = 0.1*perimeter
    approx = cv.approxPolyDP(contours[0], epsilon, True)
    approx = np.array(approx)
    approx = np.concatenate((approx, approx[:1]), axis=0)
    plt.plot(approx[:,0,0], approx[:,0,1], color='red')

    hull = cv.convexHull(contours[0])
    hull = hull[:,0,:]
    hull = np.concatenate((hull, hull[:1]), axis=0)
    plt.subplot(235)
    plt.imshow(img, cmap='gray')
    plt.plot(hull[:,0], hull[:,1], color='red')

    plt.subplot(236)
    x, y, w, h = cv.boundingRect(contours[0])
    cv.rectangle(img, (x,y), (x+w, y+h), (255,255,255), 2)
    plt.imshow(img, cmap='gray')

    plt.show()

    # aspect ratio is the ratio of width to height of bounding rectangle of the object
    aspectRatio = w/h
    # extent is the ratio of the contour area to the bounding rectangle area
    extent = area/(w*h)
    # solidity is the ratio of the contour area to the convex hull area
    solidity = area/cv.contourArea(hull) 
    # equivalent diameter is the diameter of the circle with the same area as the contour
    equivalentDiameter = np.sqrt(4*area/np.pi) 
    # orientation is the angle at which object is directed
    (x,y), (MA,ma), angle = cv.fitEllipse(contours[0])




if __name__ == '__main__':
    contour()