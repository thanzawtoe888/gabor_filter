import cv2 as cv
import numpy as np

# Create a blank image
blank = np.zeros((500, 500, 3), dtype='uint8')

# Draw green rectangle
top_left = (0, 100)
bottom_right = (500, 400)
cv.rectangle(blank, top_left, bottom_right, (0, 255, 0), thickness=2)

# Draw blue and red rectangles
cv.rectangle(blank, (100, 100), (200, 300), (255, 0, 0), cv.FILLED)  # Blue rectangle
cv.rectangle(blank, (300, 150), (400, 350), (0, 0, 255), cv.FILLED)  # Red rectangle

# Convert to HSV color space
hsv = cv.cvtColor(blank, cv.COLOR_BGR2HSV)

# Define HSV range for red color (two ranges due to red spanning both ends of the hue spectrum)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Create masks for both red ranges and combine them
mask1 = cv.inRange(hsv, lower_red1, upper_red1)
mask2 = cv.inRange(hsv, lower_red2, upper_red2)
red_mask = cv.bitwise_or(mask1, mask2)

# Apply the mask to extract only the red areas
red_detection = cv.bitwise_and(blank, blank, mask=red_mask)

# Show the images
cv.imshow('Original Image', blank)
cv.imshow('Red Mask', red_mask)
cv.imshow('Red Detection', red_detection)

cv.waitKey(0)
cv.destroyAllWindows()
