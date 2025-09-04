# distinguishing between foreground and background

#import the otsu threshold function
from skimage.filters import threshold_otsu
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

#Make the image grayscale using rgb2gray function
gray_crack = color.rgb2gray(crack_image)

#obtain the optimal threshold value using Otsu's method
thresh = threshold_otsu(gray_crack)

#apply the threshold to create a binary image
binary_crack = gray_crack > thresh

plot_comparison(crack_image, binary_crack,"OTSU Method")