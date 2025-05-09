import cv2 as cv
import numpy as np

def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)  # [1] means width, [0] means height
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def draw_grid_lines(image):
    # Draw horizontal blue lines
    cv.line(image, (0, 10), (1920, 10), (255, 0, 0), thickness=5, lineType=16)  # Blue line
    cv.line(image, (0, 125), (1920, 125), (255, 0, 0), thickness=5, lineType=16)  # Blue line
    cv.line(image, (0, 240), (1920, 240), (255, 0, 0), thickness=5, lineType=16)  # Blue line
    cv.line(image, (0, 366), (1920, 366), (255, 0, 0), thickness=5, lineType=16)  # Blue line

    # Draw vertical green lines
    cv.line(image, (85, 0), (85, 375), (0, 255, 0), thickness=5)  # Green line
    cv.line(image, (260, 0), (260, 375), (0, 255, 0), thickness=5)  # Green line
    cv.line(image, (435, 0), (435, 375), (0, 255, 0), thickness=5)  # Green line
    cv.line(image, (615, 0), (615, 375), (0, 255, 0), thickness=5)  # Green line
    cv.line(image, (795, 0), (795, 375), (0, 255, 0), thickness=5)  # Green line
    cv.line(image, (970, 0), (970, 375), (0, 255, 0), thickness=5)  # Green line
    cv.line(image, (1145, 0), (1145, 375), (0, 255, 0), thickness=5)  # Green line
    cv.line(image, (1325, 0), (1325, 375), (0, 255, 0), thickness=5)  # Green line
    cv.line(image, (1500, 0), (1500, 375), (0, 255, 0), thickness=5)  # Green line

# Load and resize the image
img = cv.imread("../images/frame_0015.jpg")
resized_img = rescale_frame(img)

# Draw grid lines on the resized image
draw_grid_lines(resized_img)

# Convert to grayscale and apply Gaussian blur
gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)

# Set the thresholds for edge classification
minVal = 50
maxVal = 150

# Apply Canny edge detection
edges = cv.Canny(blurred, minVal, maxVal)

# Display the original and edge-detected resized_img

cv.imshow('Canny Edge Detection', edges)

# Wait for a key press and close all windows
cv.waitKey(0)
cv.destroyAllWindows()

# Optional: Save the edge-detected resized_img
cv.imwrite('../results/draw_grid_canny_edge.jpg', edges)
