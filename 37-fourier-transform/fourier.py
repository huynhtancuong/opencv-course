import os 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def fourierTransform():
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imgPath, cv.IMREAD_GRAYSCALE)

    plt.figure()
    plt.subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title('Original')

    # Discrete Fourier Transform
    imgDFT = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT) # Discrete Fourier Transform
    imgDFT_DB = 20*np.log(cv.magnitude(imgDFT[:,:,0], imgDFT[:,:,1])) # Convert to dB
    plt.subplot(232)
    plt.imshow(imgDFT_DB, cmap='hot')
    plt.title('DFT')

    # Shift the zero frequency component to the center
    imgDFTShift = np.fft.fftshift(imgDFT)
    imgDFTShift_DB = 20*np.log(cv.magnitude(imgDFTShift[:,:,0], imgDFTShift[:,:,1]))
    plt.subplot(233)
    plt.imshow(imgDFTShift_DB, cmap='hot')
    plt.title('DFT Shift')

    # Create a mask, the center of the image will be 1
    r, c = img.shape[:2]
    mask = np.zeros((r, c, 2), np.uint8)
    offset = 50
    mask[int(r/2)-offset:int(r/2)+offset, int(c/2)-offset:int(c/2)+offset] = 1
    plt.subplot(234)
    plt.imshow(mask[:,:,0], cmap='gray')
    plt.title('Mask')

    # Apply the mask
    imgDFTshift_LP = imgDFTShift*mask
    imgDFTshift_LP_DB = 20*np.log(cv.magnitude(imgDFTshift_LP[:,:,0], imgDFTshift_LP[:,:,1]))
    plt.subplot(235)
    plt.imshow(imgDFTshift_LP_DB, cmap='hot')
    plt.title('DFT Shift Masked')

    # Inverse Discrete Fourier Transform
    imgInvDFT_LP = np.fft.ifftshift(imgDFTshift_LP)
    imgDFT_LP = cv.idft(imgInvDFT_LP)
    imgDFT_LP = cv.magnitude(imgDFT_LP[:,:,0], imgDFT_LP[:,:,1])
    plt.subplot(236)
    plt.imshow(imgDFT_LP, cmap='gray')
    plt.title('Inverse DFT')

    plt.show()


if __name__ == '__main__':
    fourierTransform()