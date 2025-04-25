#import the module form skimage
from skimage import data,color,filters,io
from matplotlib import pyplot as plt
import numpy as np

#load the crack image
crack_image = io.imread('../images/frame_0013.jpg')

#flip the image vertically
corrected_image_vertical = np.flipud(crack_image)

#flip the image horizontally
corrected_image_horizontal = np.fliplr(crack_image)

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
    
    
plot_comparison (crack_image,corrected_image_vertical,"Flipped vertically")
plot_comparison (crack_image,corrected_image_horizontal,"Flipped horizontally")