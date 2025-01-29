# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 14:23:28 2025

@author: luca
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

ksize = 5  #Use size that makes sense to the image and fetaure size. Large may not be good. 
#On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
sigma = 5 #Large sigma on small features will fully miss the features. 
theta = 1*np.pi/1.33  #/4 shows horizontal 3/4 shows other horizontal. Try other contributions
lamda = 1*np.pi/4  #1/4 works best for angled. 
gamma=0.8  #Value of 1 defines spherical. Calue close to 0 has high aspect ratio
#Value of 1, spherical may not be ideal as it picks up features from other regions.
phi = 1  #Phase offset. I leave it to 0. (For hidden pic use 0.8)


kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)

plt.imshow(kernel)
plt.show()

img = cv2.imread(r"images/concrete_crack.jpg")
#img = cv2.imread(r"images/concrete_crack.jpg")
#img = cv2.imread(r'C:\Users\luca\Desktop\git_repo\gabor_filter\code\images\zebra.jpg')  #Image source wikipedia: https://en.wikipedia.org/wiki/Plains_zebra
#img = cv2.imread('./images/synthetic.jpg') #USe ksize:15, s:5, q:pi/2, l:pi/4, g:0.9, phi:0.8
plt.imshow(img, cmap='gray')
plt.show()


img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)

#kernel_resized = cv2.resize(kernel, (400, 400))                    # Resize image


#plt.imshow(kernel_resized)
#plt.imshow(fimg, cmap='gray')
#plt.show()
#cv2.imshow('Kernel', kernel_resized)
#cv2.imshow('Original Img.', img)
#cv2.imshow('Filtered', fimg)
#cv2.waitKey()
#cv2.destroyAllWindows()

kernel_resized = cv2.resize(kernel, (200, 200))
# Plot results
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Display the Gabor kernel
ax[0].imshow(kernel_resized)
ax[0].set_title("Gabor Kernel")
ax[0].axis("off")

# Display the filtered image
ax[1].imshow(fimg, cmap='gray')
ax[1].set_title("Filtered Image")
ax[1].axis("off")

plt.tight_layout()
plt.show()