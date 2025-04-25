from skimage import io, exposure
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

#Apply histogram equalization to an image
def histogram_equalization(image):
    # Convert the image to grayscale
    gray_image = rgb2gray(image)
    
    # Apply histogram equalization
    equalized_image = exposure.equalize_hist(gray_image)
    
    return equalized_image

#load image
crack_image = io.imread('../images/frame_0013.jpg')


#apply standard histogram equalization
equalized_image = exposure.equalize_hist(crack_image)


# Display the original and equalized images
def plot_comparison(original, equalized, title):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 6), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(equalized, cmap=plt.cm.gray)
    ax2.set_title(title)
    ax2.axis('off')
    plt.show()
    
plot_comparison(crack_image, equalized_image, "Histogram Equalization")

