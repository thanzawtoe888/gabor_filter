#Example of applying a filter to an image
from skimage.filters import sobel
from skimage.color import rgb2gray
from skimage import io
from skimage import color
from matplotlib import pyplot as plt


# Define the show_image function
def show_image(image, title):
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
    
# Load the image
crack_image = io.imread('../images/frame_0013.jpg')

grayscale = rgb2gray(crack_image)

# Apply the Sobel filter to the grayscale image
edge_sobel = sobel(grayscale)

#Display results
show_image(crack_image, "Original Image")
show_image(edge_sobel, "Sobel Edge Detection")

