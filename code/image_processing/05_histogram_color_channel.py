#import the module form skimage
from skimage import data, color, filters, io
from matplotlib import pyplot as plt

# Modify the plot_comparison function to compare three images
def plot_comparison(original, red_hist, green_hist, blue_hist):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(16, 6), sharex=False, sharey=False)
    
    # Display the original image
    ax1.imshow(original)
    ax1.set_title('Resized_image_1920x1080')
    ax1.axis('off')
    
    # Display the red histogram
    ax2.bar(range(256), red_hist[0], color='red', alpha=0.7)
    ax2.set_title('Red Histogram')
    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Frequency')
    
    # Display the green histogram
    ax3.bar(range(256), green_hist[0], color='green', alpha=0.7)
    ax3.set_title('Green Histogram')
    ax3.set_xlabel('Pixel Intensity')
    
    # Display the blue histogram
    ax4.bar(range(256), blue_hist[0], color='blue', alpha=0.7)
    ax4.set_title('Blue Histogram')
    ax4.set_xlabel('Pixel Intensity')
    
    plt.tight_layout()
    plt.show()

# Load the image    
crack_image = io.imread('../results/rescale.jpg')  

# Extract the red, green, and blue channels from the image
red_channel = crack_image[:, :, 0]
green_channel = crack_image[:, :, 1]
blue_channel = crack_image[:, :, 2]

# Compute histograms for each channel
red_histogram = plt.hist(red_channel.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
green_histogram = plt.hist(green_channel.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
blue_histogram = plt.hist(blue_channel.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)

# Use the modified plot_comparison function
plot_comparison(crack_image, red_histogram, green_histogram, blue_histogram)