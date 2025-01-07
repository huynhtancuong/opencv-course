import cv2 as cv 
import numpy as np
import os
import matplotlib.pyplot as plt

def videoFromWebcam():
    cap = cv.VideoCapture(0)

    if not cap.isOpened():
        print('Error opening video stream or file')
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Can\'t receive frame (stream end?). Exiting ...')
            break
        cv.imshow('Webcam', frame)
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

def videoFromFile():
    root = os.getcwd()
    videoPath = os.path.join(root, 'videos', 'cat.mp4')
    cap = cv.VideoCapture(videoPath)

    while cap.isOpened():
        ret, frame = cap.read()
        cv.imshow('Video', frame)
        delay = int(1000/60) # 60 frame per second (1000 ms)
        if cv.waitKey(delay) == ord('q'):
            break

def writeVideoToFile():
    cap = cv.VideoCapture(0)

    fourcc = cv.VideoWriter_fourcc(*'XVID') # define codec 
    root = os.getcwd()
    outputPath = os.path.join(root, 'videos', 'output.avi')

    out = cv.VideoWriter(outputPath, fourcc, 20.0, (640, 480)) # 20.0 is the frame per second

    if not cap.isOpened():
        print('Error opening video stream or file')
        exit()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print('Can\'t receive frame (stream end?). Exiting ...')
            break
        out.write(frame)
        cv.imshow('Webcam', frame)
        if cv.waitKey(1) == ord('q'):
            break
    
    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # videoFromWebcam()
    # videoFromFile()
    writeVideoToFile()