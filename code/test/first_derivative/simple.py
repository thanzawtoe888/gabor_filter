import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load image in grayscale
img = cv2.imread('../../images/frame_0015.jpg', cv2.IMREAD_GRAYSCALE)

# Step 2: Select one horizontal line (e.g. middle row)
row = img[img.shape[0] // 2, :]  # 1D intensity profile

# Step 3: Apply first derivative (discrete difference)
first_derivative = np.diff(row)

# Step 4: Plot results
plt.figure(figsize=(10, 5))

plt.subplot(3,1,1)
plt.title("Original Horizontal Line (Pixel Intensities)")
plt.plot(row, color='gray')

plt.subplot(3,1,2)
plt.title("First Derivative")
plt.plot(first_derivative, color='blue')

plt.tight_layout()
plt.show()
