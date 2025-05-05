import cv2
import numpy as np
from scipy.ndimage import binary_opening

img = cv2.imread('concrete.jpg')

#Resizing
img_resized = cv2.resize(img, (256, 256))

#RGB to Grayscale Conversion
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

#edge detection sobel edge detection
sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.hypot(sobelx, sobely)
sobel = np.uint8(sobel / np.max(sobel) * 255)
_, edges = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# Bridging and spurring are not directly available in OpenCV.
# For cleaning isolated pixels:
cleaned = binary_opening(closed, structure=np.ones((2,2))).astype(np.uint8) * 255
