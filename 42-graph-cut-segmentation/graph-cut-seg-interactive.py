import cv2 as cv
import numpy as np 
import os 


class SegmentApp:
    def __init__(self, imgPath):
        self.imgPath = imgPath
        self.startX, self.startY = 0, 0
        self.buttonPressing = False
        self.tempImg = None
        self.mask = None
        self.segmentedImg = None
        self.brushSize = 3

        self.bgdModel = np.zeros((1,65), np.float64)
        self.fgdModel = np.zeros((1,65), np.float64)

    def mouseEventHandler(self, event, x, y, flags, param):
        img = param
        if event == cv.EVENT_LBUTTONDOWN:
            print("Left button pressed. Marking background")
            self.buttonPressing = 'Left'
        elif event == cv.EVENT_RBUTTONDOWN:
            print("Right button pressed. Marking foreground")
            self.buttonPressing = 'Right'
        elif event == cv.EVENT_MBUTTONDOWN:
            print("Middle button pressed. Drawing rectangle")
            self.buttonPressing = 'Mid'
            self.startX, self.startY = x, y
        elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP or event == cv.EVENT_MBUTTONUP:
            self.buttonPressing = False
            if event == cv.EVENT_MBUTTONUP:
                self.mask = np.zeros(img.shape[:2], np.uint8)
                self.rect = (self.startX, self.startY, x, y)
                cv.grabCut(img, self.mask, self.rect, self.bgdModel, self.fgdModel, iterCount=1, mode=cv.GC_INIT_WITH_RECT)
                finalMask = np.where((self.mask==2)|(self.mask==0), 0, 1).astype('uint8')
                self.segmentedImg = img*finalMask[:,:, np.newaxis]
            else:
                cv.grabCut(img, self.mask, None, self.bgdModel, self.fgdModel, iterCount=1, mode=cv.GC_INIT_WITH_MASK)
                finalMask = np.where((self.mask==2)|(self.mask==0), 0, 1).astype('uint8')
                self.segmentedImg = img*finalMask[:,:, np.newaxis]
        elif event == cv.EVENT_MOUSEMOVE and self.buttonPressing != False:
            match self.buttonPressing:
                case 'Left': # mark background
                    self.mask[y-self.brushSize:y+self.brushSize, x-self.brushSize:x+self.brushSize] = 0
                    # self.tempImg = img.copy()
                    cv.circle(self.tempImg, (x, y), self.brushSize, (0,0,255), -1)
                case 'Right': # mark foreground
                    self.mask[y-self.brushSize:y+self.brushSize, x-self.brushSize:x+self.brushSize] = 1
                    # self.tempImg = img.copy()
                    cv.circle(self.tempImg, (x, y), self.brushSize, (255,0,0), -1)
                case 'Mid':
                    # print("Middle button is pressed. Drawing rectangle")
                    self.tempImg = img.copy()
                    cv.rectangle(self.tempImg, (self.startX, self.startY), (x, y), (0,0,255), 2)

    def run(self):
        img = cv.imread(self.imgPath)
        assert img is not None, "File not found"
        
        self.mask = np.zeros(img.shape[:2], np.uint8)

        winName = 'Segment App'
        cv.namedWindow(winName)
        cv.setMouseCallback(winName, self.mouseEventHandler, img)

        while True:
            if self.buttonPressing and self.tempImg is not None:
                cv.imshow(winName, self.tempImg)
            elif self.segmentedImg is not None:
                cv.imshow(winName, self.segmentedImg)
            else:
                cv.imshow(winName, img)

            if cv.waitKey(20) == ord('q'):
                break

        cv.destroyAllWindows()




if __name__ == '__main__':
    root = os.getcwd()
    imgPath = os.path.join(root, 'images', 'cr7.jpg')
    segmentApp = SegmentApp(imgPath)
    segmentApp.run()
