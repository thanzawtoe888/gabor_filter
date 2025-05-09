import cv2 as cv
import numpy as np


resized_img = cv.imread("../results/drawed_gridline.jpg")

crack_shape = np.shape(resized_img)
print("Crack Image Shape:",crack_shape)
# cv.imshow("Resized Image", resized_img)

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
cv.imshow ("Canny Edges", edges)
cv.imshow('Measured Rectangle', resized_img)

# cv.imwrite("../results/detected_lines_.jpg", edges)  # Save the image with detected lines
cv.waitKey(0)
cv.destroyAllWindows()