
from skimage import data
from skimage import io
from skimage.filters import sobel
import numpy as np


# Load an example image from skimage
# crack_image = io.imread('../images/frame_0015.jpg', as_gray=True)
crack_image = io.imread('../images/frame_0013.jpg')


crack_shape = np.shape(crack_image)
print("Crack Image Shape:",crack_shape)

# print (crack_image.shape)
