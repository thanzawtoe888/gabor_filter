import cv2
import matplotlib.pyplot as plt

# Load the image in color
image = cv2.imread('../images/frame_0013.jpg')

# Check if the image is loaded correctly
if image is None:
    raise FileNotFoundError("Image not found or unable to load.")

# Extract the blue channel
blue = image[:, :, 2]

# Plot the histogram of the blue channel
plt.hist(blue.ravel(), bins=256)
plt.title('Histogram of Blue Channel')
plt.show()