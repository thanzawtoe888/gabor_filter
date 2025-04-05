import cv2
import numpy as np

# 1. Load the image
image = cv2.imread('../images/frame_0015.jpg')

# 2. Resize image (optional - can skip if you want full resolution)
# image = cv2.resize(image, (1280, 720))  # Uncomment if needed

# 3. Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 4. Apply Gaussian Blur (optional, helps reduce noise before edge detection)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 5. Canny Edge Detection
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

# 6. Optional: Detect Grid Lines using Hough Line Transform
#Hough transform:  to detect the list of edges in the image finds imperfect instances of objects within a class of shapes by a voting procedure adjust the Threshold parameter 
# to detect more or fewer lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
grid_line_img = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(grid_line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 7. Save output images
cv2.imwrite('../results/concrete_gray.jpg', gray)
cv2.imwrite('../results/concrete_edges.jpg', edges)

print("Preprocessing complete:")
print(" - Grayscale image saved as: /mnt/data/concrete_gray.jpg")
print(" - Canny edge image saved as: /mnt/data/concrete_edges.jpg")
