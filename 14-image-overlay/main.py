import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt

def overlayImage():
    root = os.getcwd()

    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(241)
    plt.imshow(imgRGB)
    plt.title('original')

    overlayImagePath = os.path.join(root, 'images', 'tesla.jpg')
    overlayImage = cv.imread(overlayImagePath)
    height, width = overlayImage.shape[:2]
    scale = 1/7
    height = int(height*scale)
    width = int(width*scale)
    overlayImage = cv.resize(overlayImage, (width, height), interpolation=cv.INTER_LANCZOS4)
    overlayImageRGB = cv.cvtColor(overlayImage, cv.COLOR_BGR2RGB)
    overlayImageHSV = cv.cvtColor(overlayImage, cv.COLOR_BGR2HSV)

    lower = np.array([0, 0, 0])
    upper = np.array([180, 255, 229])
    mask = cv.inRange(overlayImageHSV, lower, upper)
    plt.subplot(242)
    plt.imshow(mask, cmap='gray')
    plt.title('mask')

    y0, x0 = 180, 210
    imageRegion = imgRGB[y0:y0+height, x0:x0+width, :]
    plt.subplot(243)
    plt.title('ROI')
    plt.imshow(imageRegion)

    mask_inv = cv.bitwise_not(mask)
    # Phep AND duoc thuc hien theo tung bit trong tung pixel cua 2 anh
    imageRegionBlack = cv.bitwise_and(imageRegion, imageRegion, mask=mask_inv) # only the non-black pixels in the mask will be calculated
    plt.subplot(244)
    plt.title('Blackout logo in ROI')
    plt.imshow(imageRegionBlack)

    overlayImageForeGround = cv.bitwise_and(overlayImageRGB, overlayImageRGB, mask=mask)
    plt.subplot(245)
    plt.title('Take only region of logo from logo image ')
    plt.imshow(overlayImageForeGround)

    imageRegionLogo = cv.add(imageRegionBlack, overlayImageForeGround)
    plt.subplot(246)
    plt.title('Put logo in ROI')
    plt.imshow(imageRegionLogo)

    imgRGB[y0:y0+height, x0:x0+width, :] = imageRegionLogo
    plt.subplot(224)
    plt.imshow(imgRGB)
    plt.show()

    # Export image 
    imgBGR = cv.cvtColor(imgRGB, cv.COLOR_RGB2BGR)
    outputPath = os.path.join(root, 'images', 'cute_tesla.jpg')
    cv.imwrite(outputPath, imgBGR)



if __name__ == '__main__':
    overlayImage()