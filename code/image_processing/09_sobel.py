from matplotlib import pyplot as plt
from skimage import io, color
#Import the color module 
from skimage import color

#Import the filters module and sobel function
from skimage.filters import sobel

#load image
crack_image = io.imread('../images/frame_0013.jpg')

#Make the image grayscale
gray_crack = color.rgb2gray(crack_image)

#Apply the sobel filter to the grayscale image
edge_sobel = sobel(gray_crack)

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

plot_comparison(crack_image, edge_sobel, "Sobel Edge Detection")



