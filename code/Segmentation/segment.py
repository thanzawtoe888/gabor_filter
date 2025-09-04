import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your image
image = cv2.imread('../images/frame_0013.jpg')
# Convert from BGR (OpenCV default) to RGB for displaying
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show the image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Convert to HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define lower and upper HSV bounds for the concrete beam
lower_concrete = np.array([10, 20, 150])   # Adjust these
upper_concrete = np.array([30, 100, 255])

# Create mask
mask_concrete = cv2.inRange(image_hsv, lower_concrete, upper_concrete)

# Apply mask
concrete_segment = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_concrete)

plt.imshow(concrete_segment)
plt.title("Concrete Beam Segment")
plt.axis('off')
plt.show()
