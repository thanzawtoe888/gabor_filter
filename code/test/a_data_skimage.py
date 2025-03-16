from skimage import data
from matplotlib import pyplot as plt

rocket_image = data.rocket()
print(rocket_image.shape)  # (427, 640, 3)
plt.imshow(rocket_image)
plt.show()