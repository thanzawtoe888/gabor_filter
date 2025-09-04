# #import threshold try all function
from skimage.filters import try_all_threshold
from skimage import io, color
from matplotlib import pyplot as plt
import numpy as np

#Import the rgb to gray convertor function
from skimage.color import rgb2gray

    
# Trun the image to grayscale
def convert_to_gray(image):
    return rgb2gray(image)

# Use the try_all_threshold function to apply multiple thresholding methods
def apply_multiple_thresholds(image):
    # Convert the image to grayscale
    gray_image = convert_to_gray(image)
    
    # Apply multiple thresholding methods
    fig, ax = try_all_threshold(gray_image, figsize=(10, 8), verbose=False)
    plt.show()
    
# Load the image
crack_image = io.imread('../images/frame_0013.jpg')

# Apply the function
apply_multiple_thresholds(crack_image)


 
