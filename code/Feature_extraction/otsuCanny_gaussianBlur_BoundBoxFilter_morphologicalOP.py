# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:41:16 2025

@author: luca

✔ Better Crack Filtering: Uses Otsu’s thresholding to dynamically adjust Canny’s thresholds.
✔ Noise Reduction: Applies Gaussian blur before edge detection.
✔ Bounding Box Filtering: Ignores small objects using aspect ratio and area constraints.
✔ Morphological Operations: Reduces false positives by using dilation and erosion.
"""

import os
import random
import cv2
import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage import io
import matplotlib.pyplot as plt

# Set the directory path
positive_folder = r"D:/Crack-Dataset/5y9wdsg2zt-2/CCI/Positive"
negative_folder = r"D:/Crack-Dataset/5y9wdsg2zt-2/CCI/Negative"

# Function to randomly select an image from a folder
def random_image_from_folder(folder):
    images = os.listdir(folder)
    image_path = os.path.join(folder, random.choice(images))
    return image_path

# Function to preprocess image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian Blur to reduce noise
    return blurred

# Function to detect cracks using Canny edge detection
def detect_crack_canny(image):
    blurred = preprocess_image(image)
    
    # Extract Otsu’s threshold value correctly
    otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    
    # Define dynamic thresholds for Canny
    lower_thresh = 0.5 * otsu_thresh
    upper_thresh = 1.5 * otsu_thresh
    
    edges = cv2.Canny(blurred, int(lower_thresh), int(upper_thresh))  # Ensure integer values

    # Morphological operations to clean up edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    return edges


# Function to draw bounding boxes around detected cracks
def draw_bounding_box(image, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 50  # Minimum area to consider as a crack
    min_aspect_ratio = 2  # Minimum width-to-height ratio

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        if w * h > min_area and aspect_ratio > min_aspect_ratio:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw bounding box

    return image

# Randomly select an image from positive and negative folders
positive_image_path = random_image_from_folder(positive_folder)
negative_image_path = random_image_from_folder(negative_folder)

# Load images
positive_image = io.imread(positive_image_path)
negative_image = io.imread(negative_image_path)

# Detect cracks using Canny edge detection
positive_canny = detect_crack_canny(positive_image)
negative_canny = detect_crack_canny(negative_image)

# Draw bounding boxes
positive_image_with_box = draw_bounding_box(positive_image.copy(), positive_canny)
negative_image_with_box = draw_bounding_box(negative_image.copy(), negative_canny)

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].imshow(positive_image)
axes[0, 0].set_title('Positive Image')
axes[0, 1].imshow(positive_canny, cmap='gray')
axes[0, 1].set_title('Canny Edges')
axes[0, 2].imshow(positive_image_with_box)
axes[0, 2].set_title('Detected Cracks')

axes[1, 0].imshow(negative_image)
axes[1, 0].set_title('Negative Image')
axes[1, 1].imshow(negative_canny, cmap='gray')
axes[1, 1].set_title('Canny Edges')
axes[1, 2].imshow(negative_image_with_box)
axes[1, 2].set_title('Detected Cracks')

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
