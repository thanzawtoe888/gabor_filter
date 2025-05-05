import cv2 as cv
import numpy as np

# Create a blank 500x500 image
image = np.zeros((500, 500, 3), dtype='uint8')

# Define rectangle parameters
top_left = (100, 100)
bottom_right = (400, 400)
dot_length = 5
space_length = 5

# Draw dotted lines for the rectangle
# Top side
for x in range(top_left[0], bottom_right[0], dot_length + space_length):
    cv.line(image, (x, top_left[1]), (x + dot_length, top_left[1]), (255, 255, 255), thickness=1)

# Bottom side
for x in range(top_left[0], bottom_right[0], dot_length + space_length):
    cv.line(image, (x, bottom_right[1]), (x + dot_length, bottom_right[1]), (255, 255, 255), thickness=1)

# Left side
for y in range(top_left[1], bottom_right[1], dot_length + space_length):
    cv.line(image, (top_left[0], y), (top_left[0], y + dot_length), (255, 255, 255), thickness=1)

# Right side
for y in range(top_left[1], bottom_right[1], dot_length + space_length):
    cv.line(image, (bottom_right[0], y), (bottom_right[0], y + dot_length), (255, 255, 255), thickness=1)

# Convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv.Canny(gray, 50, 150)

# Apply Hough Line Transform
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

# Draw the detected lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)  # Green lines for detected edges

# Show the image
cv.imshow('Dotted Rectangle with Hough Transform', image)
cv.waitKey(0)
cv.destroyAllWindows()
