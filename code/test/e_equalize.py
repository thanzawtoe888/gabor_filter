#equalize refers to Histogram Equalization which is a method in image processing of contrast adjustment using the image's histogram.
#The method is useful in images with poor contrast, where the histogram is not well distributed.
# 1️⃣ Global Histogram Equalization (GHE): The classic method described above.
# 2️⃣ Adaptive Histogram Equalization (AHE): Divides the image into small regions and applies equalization locally.
# 3️⃣ Contrast Limited Adaptive Histogram Equalization (CLAHE): AHE but with contrast limiting to prevent over-enhancing noise in uniform regions.

# 👉 Example: For crack detection, CLAHE often works better to avoid over-boosting the background noise.
import cv2
import matplotlib.pyplot as plt

# Load the image in grayscale
img = cv2.imread('../images/frame_0013.jpg', 0)

# Apply Histogram Equalization
equalized_img = cv2.equalizeHist(img)

# Display original vs. equalized image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(equalized_img, cmap='gray')

plt.show()
