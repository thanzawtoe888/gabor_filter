#import the module form skimage
from skimage import data,color,filters,io
from matplotlib import pyplot as plt
import numpy as np

    
#load the image    
crack_image = io.imread('../images/frame_0013.jpg')  

#Extract the red channel from the image
red_channel = crack_image[:, :, 0]

#display the red channgel of the image in grayscale, showing the intensity fo red across the image
plt.figure(figsize=(8, 6))
plt.imshow(red_channel, cmap='gray')
plt.title('Red Channel')    
plt.show() 