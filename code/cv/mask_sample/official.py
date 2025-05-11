import cv2 as cv
import numpy as np
 


 # Example usage
def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] // 5)  # [1] means width, [0] means height
    height = int(frame.shape[0] // 5)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Load and resize the image
img = cv.imread("../../images/frame_0013.jpg")
resized_img = rescale_frame(img)
    # Convert BGR to HSV
hsv = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)

    # define range of blue color in HSV
lower_blue = np.array([107,19,235])
upper_blue = np.array([127,39,255])
 
    # Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)
 
    # Bitwise-AND mask and original image
res = cv.bitwise_and(resized_img,resized_img, mask= mask)
 
cv.imshow('frame',resized_img)
cv.imshow('mask',mask)
cv.imshow('res',res)

   
cv.waitKey(0)
cv.destroyAllWindows()