import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = '../images/frame_0013.jpg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# Define the four points of the bounding box (pixel coordinates)
points = [(0.6, 775), (3833, 775), (0.6, 1496), (3833, 1496)]

# Convert points to a numpy array
pts = np.array(points, np.int32)
pts = pts.reshape((-1, 1, 2))

# Create a mask for the bounding box
mask = np.zeros_like(image[:, :, 0])
cv2.fillPoly(mask, [pts], 255)

# Extract the region of interest (ROI) using the mask
roi = cv2.bitwise_and(image, image, mask=mask)

# Convert ROI to grayscale
gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Apply edge detection (Canny)
crack_edges = cv2.Canny(gray_roi, 50, 150)

# Find contours of the crack edges
contours, _ = cv2.findContours(crack_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Define base_line_y
base_line_y = max(points, key=lambda item: item[1])[1]

# Filter contours that start near the bottom line of the rectangle
filtered_contours = []
for cnt in contours:
    for point in cnt[:, 0, :]:
        if point[1] >= base_line_y - 10:  # Adjust the threshold as needed
            filtered_contours.append(cnt)
            break

# Draw the filtered contours on the original image
cv2.drawContours(image, filtered_contours, -1, (0, 0, 255), 2)

# Overlay crack edges onto the original image
image[crack_edges > 0] = [0, 0, 255]  # Highlight cracks in red

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the output image
cv2.imwrite('output_with_crack_detection.png', image)
