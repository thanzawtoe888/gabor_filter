import cv2 as cv
import numpy as np

# Create a blank image
image = np.zeros((500, 500, 3), dtype="uint8")

# Draw a horizontal line
cv.line(image, (50, 250), (450, 250), (255, 0, 0), thickness=2)  # Blue line

# Draw a vertical line
cv.line(image, (250, 50), (250, 450), (0, 255, 0), thickness=2)  # Green line

# Draw a diagonal line
cv.line(image, (50, 50), (450, 450), (0, 0, 255), thickness=2)  # Red line

# Draw a "crack" line (custom zigzag pattern)
points = [(50, 400), (150, 300), (250, 400), (350, 300), (450, 400)]
for i in range(len(points) - 1):
    cv.line(image, points[i], points[i + 1], (255, 255, 0), thickness=2)  # Yellow line

# Display the image
cv.imshow("Lines", image)

cv.waitKey(0)
cv.destroyAllWindows()