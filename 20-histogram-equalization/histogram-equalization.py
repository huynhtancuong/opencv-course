import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def histogramEqual():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'bad.jpg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdfNorm = cdf * float(hist.max()) / cdf.max()

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm, color='b')
    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixels')

    # cdf_m = np.ma.masked_equal(cdf,0)
    # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    # cdf = np.ma.filled(cdf_m,0).astype('uint8')
    
    # equImg = cdf[img]

    equImg = cv.equalizeHist(img)
    equHist = cv.calcHist(equImg, [0], None, [256], [0, 256])
    equcdf = equHist.cumsum()
    equacdfNorm = equcdf * float(equHist.max()) / equcdf.max()

    plt.subplot(232)
    plt.imshow(equImg, cmap='gray')
    plt.title('Hist Equalization')
    plt.subplot(235)
    plt.plot(equHist)
    plt.plot(equacdfNorm, color='b')


    claheObj = cv.createCLAHE(clipLimit=5, tileGridSize=(8,8))
    claheImg = claheObj.apply(img)
    claheHist = cv.calcHist(claheImg, [0], None, [256], [0, 256])
    clahecdf = claheHist.cumsum()
    clahecdfNorm = clahecdf * float(claheHist.max()) / clahecdf.max()

    plt.subplot(233)
    plt.imshow(claheImg, cmap='gray')
    plt.title('CLAHE')
    plt.subplot(236)
    plt.plot(claheHist)
    plt.plot(clahecdfNorm, color='b')

    plt.show()


if __name__ == '__main__':
    histogramEqual()