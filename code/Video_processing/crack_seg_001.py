import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '../images/frame_0013.jpg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# Define the bounding box coordinates
points = [(0, 775), (3833, 775), (3833, 1496), (0, 1496)]
pts = np.array(points, np.int32).reshape((-1, 1, 2))

# Create a mask for the bounding box
mask = np.zeros_like(image)
cv2.fillPoly(mask, [pts], (255, 255, 255))
masked_image = cv2.bitwise_and(image, mask)

# Convert to grayscale and apply adaptive thresholding
gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Apply Otsu's thresholding
_, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Use Otsu's thresholding result for further processing
thresh = otsu_thresh

# Morphological operations to enhance cracks
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Edge detection
edges = cv2.Canny(closing, 50, 150)

# Draw the bounding box on the original image
cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

# Find contours and filter them
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
base_line_y = 1496

# Initialize a counter for cracks
crack_count = 0

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    if 0.1 < aspect_ratio < 10.0 and y + h < base_line_y - 10:
        # Overlay crack lines by drawing the contour with a distinct color (e.g., green)
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
        # Draw a bounding box around the crack
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        # Increment the crack counter
        crack_count += 1

# Display the crack count
print(f"Number of cracks detected: {crack_count}")

# Overlay the crack count on the image
cv2.putText(image, f"Cracks: {crack_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 255, 255), 2, cv2.LINE_AA)

# Display results
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the output image
cv2.imwrite('crack_segmentation_output.png', image)
