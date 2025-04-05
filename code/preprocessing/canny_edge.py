import cv2
import numpy as np

# Load the image
image = cv2.imread('../images/frame_0015.jpg')
if image is None:
    raise FileNotFoundError("Image not found or unable to load.")

# Step 1: Use the whole image as Region of Interest
roi = image

# Step 2: Convert RGB to Grayscale
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Step 3: Apply Gaussian Blur (optional, but helps with noise)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Canny Edge Detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# Optional Step 5: Detect Grid Lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
grid_line_img = roi.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(grid_line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# # Save or display
# cv2.imwrite('gray.jpg', gray)
# cv2.imwrite('edges.jpg', edges)
# cv2.imwrite('grid_lines.jpg', grid_line_img)

# cv2.imshow("Gray", gray)
# cv2.imshow("Edges", edges)
# cv2.imshow("Grid Lines", grid_line_img)

# display the roi  

cv2.imshow("Region of Interest", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
