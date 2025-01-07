import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt

def drawCircle(event, x, y, flags, param):
    img = param
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img, center=(x,y), radius=50, color=(255,0,0), thickness=2)

def doubleClickDrawing():
    root = os.getcwd()
    imagePath = os.path.join(root, 'images', 'cute.jpeg')  
    img = cv.imread(imagePath)
    
    windowName = 'Drawing app'
    cv.namedWindow(windowName)
    cv.setMouseCallback(windowName, drawCircle, img)

    while True:
        cv.imshow(windowName, img)
        if cv.waitKey(20) == ord('q'):
            cv.destroyAllWindows()
            break

class DrawingApp:
    def __init__(self, imgPath):
        self.imgPath = imgPath
        self.startX, self.startY = 0, 0
        self.drawing = False
        self.tempImg = None

    def drawLine(self, event, x, y, flags, param):
        img = param
        if event == cv.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.startX, self.startY = x, y
        elif event == cv.EVENT_MOUSEMOVE and self.drawing:
            self.tempImg = img.copy()
            cv.line(self.tempImg, pt1=(self.startX, self.startY), pt2=(x, y), color=(255,0,0), thickness=2)
        elif event == cv.EVENT_LBUTTONUP:
            cv.line(img, pt1=(self.startX, self.startY), pt2=(x, y), color=(255,0,0), thickness=2)
            self.drawing = False        

    def run(self):
        img = cv.imread(self.imgPath)
        windowName = 'Drawing app'
        cv.namedWindow(windowName)
        cv.setMouseCallback(windowName, self.drawLine, img)

        while True:
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
    # doubleClickDrawing()
    holdAndDragDrawing()