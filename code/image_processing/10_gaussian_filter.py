from matplotlib import pyplot as plt
from skimage import io
#Import Gaussian Filter
from skimage.filters import gaussian

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


#load image
crack_image = io.imread('../images/frame_0013.jpg')


#Apply filter
gaussian_image = gaussian(crack_image, sigma=1, channel_axis=-1)

# Show original and resulting image to compare
plot_comparison(crack_image, gaussian_image, "Gaussian Filter")