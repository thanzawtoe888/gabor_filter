import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = '../images/frame_0013.jpg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# Define the four points of the bounding box
points = [(0, 775), (3833, 775), (3833, 1496), (0, 1496)]

# Convert points to a numpy array
pts = np.array(points, np.int32)
pts = pts.reshape((-1, 1, 2))

# Draw the bounding box
cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

# Create a mask for the bounding box
mask = np.zeros_like(image)
cv2.fillPoly(mask, [pts], (255, 255, 255))
masked_image = cv2.bitwise_and(image, mask)

# Detect the crack only inside the bounding box
gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

# Find contours to detect curved cracks
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

base_line_y = 1496
for contour in contours:
    # Filter out small rectangular contours
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if aspect_ratio < 0.2 or aspect_ratio > 5.0:  # Adjust aspect ratio threshold to detect smaller cracks
        # Check if any point in the contour is near the base line
        if any(point[0][1] >= base_line_y - 50 for point in contour):
            # Filter out noise near the base line
            if y + h < base_line_y - 10:
                cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the output image
cv2.imwrite('output_with_crack_detection.png', image)