import numpy as np
from skimage import color
from skimage import data
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self):
        self.original, self.coffee_image, self.coin_image = self.load_images()
        self.grayscale = self.convert_to_grayscale(self.original)

    def load_images(self):
        original = data.astronaut()
        coffee_image = data.coffee()
        coin_image = data.coins()
        return original, coffee_image, coin_image

    def convert_to_grayscale(self, image):
        return color.rgb2gray(image)

    def show_image(self, image, title='Image', cmap='gray'):
        plt.imshow(image, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def show_comparison(self, images, titles, cmap='gray'):
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap=cmap if len(img.shape) == 2 else None)
            ax.set_title(title)
            ax.axis('off')
        plt.show()

    def display_images(self):
        self.show_image(self.grayscale, 'Original in Grayscale')
        self.show_image(self.original, 'Original in Color', cmap=None)
        self.show_image(self.coffee_image, 'Coffee in Color', cmap=None)
        self.show_image(self.coin_image, 'Coin in Grayscale')
        
    def extract_rgb(self, image):
        red_channel = image[:, :, 0]    # Red channel
        green_channel = image[:, :, 1]  # Green channel
        blue_channel = image[:, :, 2]   # Blue channel
        return red_channel, green_channel, blue_channel
    
    def display_comparison(self):
        images = [self.original, self.grayscale, self.coffee_image, self.coin_image]
        titles = ['Original in Color', 'Original in Grayscale', 'Coffee in Color', 'Coin in Grayscale']
        self.show_comparison(images, titles)
        
    def compare_red_channel(self):
        red_channel, _, _ = self.extract_rgb(self.original)
        self.show_comparison([red_channel, self.grayscale], ['Red Channel', 'Grayscale'])

    def compare_green_channel(self):
        _, green_channel, _ = self.extract_rgb(self.original)
        self.show_comparison([green_channel, self.grayscale], ['Green Channel', 'Grayscale'])

    def compare_blue_channel(self):
        _, _, blue_channel = self.extract_rgb(self.original)
        self.show_comparison([blue_channel, self.grayscale], ['Blue Channel', 'Grayscale'])

    def compare_all_channels(self):
        red_channel, green_channel, blue_channel = self.extract_rgb(self.original)
        self.show_comparison(
            [red_channel, green_channel, blue_channel, self.original],
            ['Red Channel', 'Green Channel', 'Blue Channel', 'Grayscale']
        )

def main():
    processor = ImageProcessor()    
    # processor.display_images()
    # processor.display_comparison()
    processor.extract_rgb(processor.original)
    # processor.compare_red_channel()
    # processor.compare_green_channel()
    # processor.compare_blue_channel()
    processor.compare_all_channels()
    
if __name__ == "__main__":
    main()

