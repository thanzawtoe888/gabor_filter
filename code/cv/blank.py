import cv2 as cv
import numpy as np

def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)  # [1] means width, [0] means height
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Load and resize the image
img = cv.imread("../images/frame_0015.jpg")
resized_img = rescale_frame(img)


cv.line(resized_img, (0,10), (1920, 10), (255, 0, 0), thickness=5,lineType=16)  # Blue line

cv.line(resized_img, (0,125), (1920, 125), (255, 0, 0), thickness=5,lineType=16)  # Blue line

cv.line(resized_img, (0,240), (1920, 240), (255, 0, 0), thickness=5,lineType=16)  # Blue line

cv.line(resized_img, (0,366), (1920, 366), (255, 0, 0), thickness=5,lineType=16)  # Blue line

cv.line(resized_img, (85, 0), (85, 375), (0, 255, 0), thickness=5)  # Green line

cv.line(resized_img, (260, 0), (260, 375), (0, 255, 0), thickness=5)  # Green line

cv.line(resized_img, (435, 0), (435, 375), (0, 255, 0), thickness=5)  # Green line

cv.line(resized_img, (615, 0), (615, 375), (0, 255, 0), thickness=5)  # Green line

cv.line(resized_img, (795, 0), (795, 375), (0, 255, 0), thickness=5)  # Green line

cv.line(resized_img, (970, 0), (970, 375), (0, 255, 0), thickness=5)  # Green line

cv.line(resized_img, (1145, 0), (1145, 375), (0, 255, 0), thickness=5)  # Green line

cv.line(resized_img, (1325, 0), (1325, 375), (0, 255, 0), thickness=5)  # Green line

cv.line(resized_img, (1500, 0), (1500, 375), (0, 255, 0), thickness=5)  # Green line

cv.line(resized_img, (1675, 0), (1675, 375), (0, 255, 0), thickness=5)  # Green line

cv.line(resized_img, (1848, 0), (1848, 375), (0, 255, 0), thickness=5)  # Green line

cv.imshow('Draw Gridlines', resized_img)

cv.imwrite("../results/drawed_gridline.jpg", resized_img)  # Save the image with detected lines
cv.waitKey(0)
cv.destroyAllWindows()
