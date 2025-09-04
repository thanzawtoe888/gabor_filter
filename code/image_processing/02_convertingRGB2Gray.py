#import the module form skimage
from skimage import data,color,filters,io
from matplotlib import pyplot as plt

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

#load the rocket image
crack_image = io.imread('../images/frame_0013.jpg')


# Convert the image to grayscale
gray_crack = color.rgb2gray(crack_image)

plot_comparison (crack_image,gray_crack,"changed2Grayscale")
  

