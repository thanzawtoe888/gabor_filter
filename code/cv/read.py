import cv2 as cv
import numpy as np


capture = cv.VideoCapture(0)

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()