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

# Define HSV range for red color (two ranges due to red spanning both ends of the hue spectrum)
lower_red1 = np.array([0, 19, 80])
upper_red1 = np.array([20, 50, 255])

# Create masks for both red ranges and combine them
mask1 = cv.inRange(hsv, lower_red1, upper_red1)
# mask2 = cv.inRange(hsv, lower_red2, upper_red2)
# calcline_mask = cv.bitwise_or(mask1, mask2)

# Apply the mask to extract only the red areas
calcline_detection = cv.bitwise_and(resized_img, resized_img, mask=mask1)

# Show the images
cv.imshow('Original Image', resized_img)
cv.imshow('HSV Image', hsv)
cv.imshow('Calcline Mask', mask1)
cv.imshow('Calcline Detection', calcline_detection)

cv.imwrite('HSV_Image.jpg', hsv)
cv.imwrite('calcline_mask.jpg', calcline_detection)
cv.imwrite('calcline_detection.jpg', calcline_detection)
cv.imwrite('Calcline_mask.jpg', mask1)
cv.waitKey(0)
cv.destroyAllWindows()
