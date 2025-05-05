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
cv.imshow("Resized Image", resized_img)

# Convert to grayscale and apply Gaussian blur
gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv.Canny(blurred, threshold1=50, threshold2=150)

# Apply Hough Line Transform
lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Draw the detected lines on the image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(resized_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)  # Green lines for detected edges

# Display the image with detected lines
cv.imshow('Measured Rectangle', resized_img)

cv.waitKey(0)
cv.destroyAllWindows()