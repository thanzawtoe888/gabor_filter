import cv2 as cv
import numpy as np

# Create a blank 500x500 image
image = np.zeros((500, 500, 3), dtype='uint8')

# Draw a rectangle
cv.rectangle(image, (50, 50), (200, 150), (255, 0, 0), thickness=2)  # Blue rectangle

# Draw a cylinder (approximated as two circles connected by lines)
cv.circle(image, (300, 100), 50, (0, 255, 0), thickness=2)  # Top circle
cv.circle(image, (300, 200), 50, (0, 255, 0), thickness=2)  # Bottom circle
cv.line(image, (250, 100), (250, 200), (0, 255, 0), thickness=2)  # Left line
cv.line(image, (350, 100), (350, 200), (0, 255, 0), thickness=2)  # Right line

# Draw a circle
cv.circle(image, (100, 300), 50, (0, 0, 255), thickness=2)  # Red circle

# Draw a triangle
pts = np.array([[300, 300], [250, 400], [350, 400]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv.polylines(image, [pts], isClosed=True, color=(255, 255, 0), thickness=2)  # Yellow triangle

# Convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv.Canny(gray, 50, 150)

# Apply Hough Line Transform
lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

# Initialize variables to store the rectangle's bounding box
x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0

# Draw the detected lines and find the rectangle's bounding box
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)  # Green lines for detected edges
        x_min = min(x_min, x1, x2)
        y_min = min(y_min, y1, y2)
        x_max = max(x_max, x1, x2)
        y_max = max(y_max, y1, y2)

# Crop the ROI (rectangle)
if x_min < x_max and y_min < y_max:  # Ensure valid bounding box
    roi = image[y_min:y_max, x_min:x_max]
    cv.imshow("Cropped ROI", roi)

# Display the results
cv.imshow("Original Image with Detected Lines", image)
cv.imshow("Edges", edges)

cv.waitKey(0)
cv.destroyAllWindows()