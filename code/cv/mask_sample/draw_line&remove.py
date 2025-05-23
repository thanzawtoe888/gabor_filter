import cv2 as cv
import numpy as np

def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Load and resize the image
img = cv.imread("../../images/frame_0015.jpg")
resized_img = rescale_frame(img)

# Create a mask with the same dimensions as the resized image
mask = np.zeros_like(resized_img)

# Draw lines on the mask
line_color = (255, 255, 255)  # White mask
thickness = 5

# Blue horizontal lines
cv.line(mask, (0, 10), (1920, 10), line_color, thickness, lineType=16)
cv.line(mask, (0, 125), (1920, 125), line_color, thickness, lineType=16)
cv.line(mask, (0, 240), (1920, 240), line_color, thickness, lineType=16)
cv.line(mask, (0, 366), (1920, 366), line_color, thickness, lineType=16)

# Green vertical lines
x_coords = [85, 260, 435, 615, 795, 970, 1145, 1325, 1500, 1675, 1848]
for x in x_coords:
    cv.line(mask, (x, 0), (x, 375), line_color, thickness)

# Apply the mask to the resized image
masked_img = cv.bitwise_and(resized_img, cv.bitwise_not(mask))

# Convert to grayscale for Canny edge detection
gray_img = cv.cvtColor(masked_img, cv.COLOR_BGR2GRAY)

# Apply Canny edge detection
lower_threshold = 50
upper_threshold = 150
edges = cv.Canny(gray_img, lower_threshold, upper_threshold)

# Display the Canny edges
cv.imshow('Canny Edge Detection', edges)

# Save the edge-detected image
cv.imwrite("../../results/canny_edges.jpg", edges)

# Wait and close windows
cv.waitKey(0)
cv.destroyAllWindows()
