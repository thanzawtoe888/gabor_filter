import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = '../images/frame_0013.jpg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# Define the four points of the bounding box (pixel coordinates)
points = [(0, 775), (3833, 775), (3833, 1496), (0, 1496)]

# Convert points to a numpy array
pts = np.array(points, np.int32)
pts = pts.reshape((-1, 1, 2))

# Draw the bounding box
cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

# Draw the middle horizontal line
middle_y = height // 2
cv2.line(image, (0, middle_y), (width, middle_y), (0, 0, 255), 2)

# Create a mask for the bounding box
mask = np.zeros_like(image[:, :, 0])
cv2.fillPoly(mask, [pts], 255)

# Extract the region of interest (ROI) using the mask
roi = cv2.bitwise_and(image, image, mask=mask)

# Convert ROI to grayscale
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Apply edge detection (Canny)
crack_edges = cv2.Canny(gray_roi, 50, 150)

# Overlay crack edges onto the original image
image[crack_edges > 0] = [0, 0, 255]  # Highlight cracks in red

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the output image
cv2.imwrite('output_with_crack_detection.png', image)
