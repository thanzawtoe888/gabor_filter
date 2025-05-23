import cv2 as cv
import numpy as np


def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] // 5)  # [1] means width, [0] means height
    height = int(frame.shape[0] // 5)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


# Load and resize the image
img = cv.imread("../../images/frame_0013.jpg")
resized_img = rescale_frame(img)

# Convert to HSV color space
hsv = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)

# Apply histogram equalization to the V (value) channel
h, s, v = cv.split(hsv)
v_eq = cv.equalizeHist(v)
hsv_eq = cv.merge([h, s, v_eq])

# Convert back to BGR color space
resized_img_eq = cv.cvtColor(hsv_eq, cv.COLOR_HSV2BGR)

# Define HSV range for calcined clay
lower_calcined = np.array([0, 5, 80])
upper_calcined = np.array([20, 255, 255])

# Create a mask for the calcined clay
mask = cv.inRange(hsv_eq, lower_calcined, upper_calcined)

# Apply the mask to the original image
calcined_mask = cv.bitwise_and(resized_img_eq, resized_img_eq, mask=mask)

# Display the results
cv.imshow("Original Image", resized_img)
cv.imshow("Histogram Equalized Image", resized_img_eq)
cv.imshow("Calcined Concrete Mask", mask)
cv.imshow("Calcined Concrete Extracted", calcined_mask)

# Wait for a key press and close all windows
cv.waitKey(0)
cv.destroyAllWindows()