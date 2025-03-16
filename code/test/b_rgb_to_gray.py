############ Modular code structure ############
from skimage import data, color
import matplotlib.pyplot as plt

def load_image():
    return data.rocket()

def convert_to_gray(image):
    return color.rgb2gray(image)

def show_image(image, title='Image', cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():
    # Load the rocket image from skimage library
    rocket_image = load_image()

    # Convert the RGB image to grayscale
    gray_image = convert_to_gray(rocket_image)

    # Display the grayscale image
    show_image(gray_image, title='Grayscale Rocket Image')

if __name__ == "__main__":
    main()
