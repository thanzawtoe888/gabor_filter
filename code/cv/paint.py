import cv2 as cv
import numpy as np

blank = np.zeros((500, 500,3), dtype='uint8')
# cv.imshow('Blank', blank)

# # paint the image a certain color
# blank[:] = 0,255,0
# cv.imshow('Green', blank)

# cv.rectangle(blank, (0,0), (400,250), (0,255,0), thickness=2)

## Draw grid lines every 100 pixels
for i in range(0, blank.shape[1], 100):  # Vertical lines
    cv.line(blank, (i, 0), (i, blank.shape[0]), color=(200, 200, 200), thickness=1)
for j in range(0, blank.shape[0], 100):  # Horizontal lines
    cv.line(blank, (0, j), (blank.shape[1], j), color=(200, 200, 200), thickness=1)

# ## Mark the point (0, 0)
# cv.circle(blank, (0, 0), radius=5, color=(0, 0, 255), thickness=-1)
# ## Mark the point (400, 250)
# cv.circle(blank, (400, 250), radius=5, color=(255, 0, 0), thickness=-1)

## Convert the image to grayscale
gray = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)

## Apply edge detection
edges = cv.Canny(gray, 50, 150)

## Detect lines using Hough Line Transform
lines = cv.HoughLines(edges, 1, np.pi / 180, 100)

## Count the number of lines detected
line_count = len(lines) if lines is not None else 0
print(f"Number of lines detected: {line_count}")

if lines is not None:
    for idx, line in enumerate(lines):
        rho, theta = line[0]  # Extract rho and theta
        print(f"Line {idx + 1}: rho = {rho}, theta = {theta}")

cv.imshow('Rectangle', blank)
cv.waitKey(0)
