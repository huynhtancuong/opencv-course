import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt

class DrawingApp:
    def __init__(self, imgPath):
        self.imgPath = imgPath
        self.startX, self.startY = 0, 0
        self.drawing = False
        self.tempImg = None

    def drawLine(self, event, x, y, flags, param):
        img = param
        color = (self.b, self.g, self.r)
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.startX, self.startY = x, y
        elif event == cv.EVENT_MOUSEMOVE and self.drawing:
            self.tempImg = img.copy()
            cv.line(self.tempImg, pt1=(self.startX, self.startY), pt2=(x, y), color=color, thickness=2)
        elif event == cv.EVENT_LBUTTONUP:
            cv.line(img, pt1=(self.startX, self.startY), pt2=(x, y), color=color, thickness=2)
            self.drawing = False        

    def run(self):
        img = cv.imread(self.imgPath)
        windowName = 'Drawing app'
        cv.namedWindow(windowName)
        cv.setMouseCallback(windowName, self.drawLine, img)
        cv.createTrackbar('B', windowName, 0, 255, lambda x: None)
        cv.createTrackbar('G', windowName, 0, 255, lambda x: None)
        cv.createTrackbar('R', windowName, 0, 255, lambda x: None)

        while True:
            self.b = cv.getTrackbarPos('B', windowName)
            self.g = cv.getTrackbarPos('G', windowName)
            self.r = cv.getTrackbarPos('R', windowName)

            if self.drawing and self.tempImg is not None:
                cv.imshow(windowName, self.tempImg)
            else:
                cv.imshow(windowName, img)
            if cv.waitKey(20) == ord('q'):
                cv.destroyAllWindows()
                break

def holdAndDragDrawing():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')  
    drawingApp = DrawingApp(imagePath)
    drawingApp.run()
    

if __name__ == '__main__':
    holdAndDragDrawing()