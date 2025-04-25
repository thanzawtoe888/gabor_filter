# local ->> for the non-uniform background
# global ->> for the uniform background

# #import threshold local function
from skimage.filters import threshold_local
from skimage import io, color
from matplotlib import pyplot as plt
import numpy as np

#show the original image 
def plot_comparison (original, filtered, titile_filtered):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 6) , sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(titile_filtered)
    ax2.axis('off')
    plt.show()

#load the image
crack_image = io.imread('../images/frame_0013.jpg') 

#turn the image to grayscale
gray_crack = color.rgb2gray(crack_image)

#set block size and offset
block_size = 35
offset = 10

#Apply $$LOCAL$$ thresholding
local_thresh = threshold_local(gray_crack, block_size, offset=offset)
binary_local = gray_crack > local_thresh


plot_comparison(crack_image, binary_local,"Local Thresholding")
