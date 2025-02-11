import os
import random
import cv2
import numpy as np
from skimage.feature import hog, canny
from skimage.color import rgb2gray
from skimage import io

import matplotlib.pyplot as plt


# Set the directory path
directory = r"D:/Crack-Dataset/5y9wdsg2zt-2/crack-detect/CCI"

positive_folder = os.path.join(r"D:/Crack-Dataset/5y9wdsg2zt-2/CCI/Positive")
negative_folder = os.path.join(r"D:/Crack-Dataset/5y9wdsg2zt-2/CCI/Negative")

# Function to randomly select an image from a folder
def random_image_from_folder(folder):
    images = os.listdir(folder)
    image_path = os.path.join(folder, random.choice(images))
    return image_path

# Function to detect cracks using HOG
def detect_crack_hog(image):
    gray_image = rgb2gray(image)
    fd, hog_image = hog(gray_image, visualize=True, block_norm='L2-Hys')
    return hog_image

# Function to detect cracks using Canny edge detection
def detect_crack_canny(image):
    gray_image = rgb2gray(image)
    edges = canny(gray_image)
    return edges

# Function to draw bounding box around detected cracks
def draw_bounding_box(image, edges):
    contours, _ = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

# Randomly select an image from positive and negative folders
positive_image_path = random_image_from_folder(positive_folder)
negative_image_path = random_image_from_folder(negative_folder)

# Load images
positive_image = io.imread(positive_image_path)
negative_image = io.imread(negative_image_path)

# Detect cracks using HOG
positive_hog = detect_crack_hog(positive_image)
negative_hog = detect_crack_hog(negative_image)

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
axes[0, 1].imshow(positive_hog, cmap='gray')
axes[0, 1].set_title('HOG Features')
axes[0, 2].imshow(positive_image_with_box)
axes[0, 2].set_title('Canny with Bounding Box')

axes[1, 0].imshow(negative_image)
axes[1, 0].set_title('Negative Image')
axes[1, 1].imshow(negative_hog, cmap='gray')
axes[1, 1].set_title('HOG Features')
axes[1, 2].imshow(negative_image_with_box)
axes[1, 2].set_title('Canny with Bounding Box')

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()