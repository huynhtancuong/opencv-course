import cv2 as cv 
import os
import numpy as np
import matplotlib.pyplot as plt

def gaussianKernel(size, sigma):
    kernel = cv.getGaussianKernel(size, sigma) # 1D kernel
    kernel = np.outer(kernel, kernel.transpose()) # 2D kernel
    return kernel

def gaussianFilter():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imgPath)   
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    noisyImg = img.copy()
    noiseProb = 0.05
    noise = np.random.rand(noisyImg.shape[0], noisyImg.shape[1])
    noisyImg[noise < noiseProb/2] = 0
    noisyImg[noise > 1 - noiseProb/2] = 255

    n = 51
    kernel = gaussianKernel(n, 8)
    
    # Plot kernel with matplotlib
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(kernel)

    ax = fig.add_subplot(122, projection='3d')
    x = np.arange(0, n, 1)
    y = np.arange(0, n, 1)
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, kernel, cmap='viridis')
    plt.show()

    # Plot image with OpenCV
    winName = 'Gaussian Filter'
    cv.namedWindow(winName)
    cv.createTrackbar('sigma', winName, 1, 20, lambda x: None)
    cv.createTrackbar('n', winName, 1, 20, lambda x: None)
    height, width, _ = noisyImg.shape
    scale = 1
    width = int(width*scale)
    height = int(height*scale)
    noisyImg = cv.resize(noisyImg, (width, height), interpolation=cv.INTER_LANCZOS4)

    while True:
        sigma = cv.getTrackbarPos('sigma', winName)
        n = cv.getTrackbarPos('n', winName)
        n = 1 if n < 1 else n
        kernel = gaussianKernel(n, sigma)
        imgFilter = cv.filter2D(noisyImg, -1, kernel)
        cv.imshow(winName, imgFilter)
        if cv.waitKey(20) == ord('q'):
            break

    cv.destroyAllWindows()



if __name__ == '__main__':
    gaussianFilter()