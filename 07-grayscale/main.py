import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt

def grayscale():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')
    img = cv.imread(imagePath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    b,g,r = cv.split(img)

    myGray = 0.114*b + 0.587*g + 0.299*r

    plt.figure()
    plt.subplot(121)
    plt.imshow(gray, cmap='gray')
    plt.title('OpenCV Grayscale')

    plt.subplot(122)
    plt.imshow(myGray, cmap='gray')
    plt.title('My Grayscale')

    plt.show()

    total_square_error = np.sum((gray - myGray)**2)

    print('Total square error: ', total_square_error)
    # The total square error is not zero because of the floating point precision error


if __name__ == '__main__':
    grayscale()