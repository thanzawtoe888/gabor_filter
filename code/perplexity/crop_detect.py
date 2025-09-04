import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image
img = cv2.imread('../images/frame_0013.jpg')
# Convert to HSV for color thresholding
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define HSV range for beige concrete (approximate)
lower = np.array([10, 20, 150])
upper = np.array([30, 80, 255])

# Create mask and clean up
mask = cv2.inRange(hsv, lower, upper)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# Find contours and bounding box
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # Find largest contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    crop = img[y:y+h, x:x+w]
else:
    crop = img.copy()
    x, y, w, h = 0, 0, img.shape[1], img.shape[0]

# Crack detection in crop
gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)

# Display results
plt.figure(figsize=(12,6))
plt.subplot(1,3,1); plt.title('Original Image'); plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.subplot(1,3,2); plt.title('Cropped Specimen'); plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.subplot(1,3,3); plt.title('Detected Cracks'); plt.imshow(edges, cmap='gray'); plt.axis('off')
plt.tight_layout()
plt.show()