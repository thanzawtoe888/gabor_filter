import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2

## 1. Load the image
## read the image using OpenCv
image = cv2.imread('../../images/frame_0013.jpg')  ## your location
## show the image 
plt.imshow(image)

##2. Convert to BGR to RGB
## convert the image from BGR to RGB format
## OpenCV reads images in BGR format, while matplotlib expects RGB format.
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)

##3. Convert to RGB to HSV
## use the covtColor function again 
hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
plt.imshow(hsv)
## show the image in HSV format

##4.  gaussian blur to the hsv image
## use the GaussianBlur function to apply the blur
# blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
# plt.imshow(blurred)

##5. Build the Mask
## first set the values of the minimum and maximum range and use cv2 ## inRange function 
# mini = np.array([3,25,200]) 
# maxi = np.array([10,45,250])
mini = np.array([255,255,0])
maxi = np.array([255,255, 102])
## use the inRange function to build the mask 
mask1 = cv2.inRange(hsv, mini, maxi)
## Show the mask 
plt.imshow(mask1)


##6. use cvtColor to change the color format
rgb_mask = cv2.cvtColor(mask1, cv2.COLOR_GRAY2RGB)
## applying mask onto the RGB image
img = cv2.addWeighted(rgb_mask, 0.5, rgb, 0.5, 0)
plt.imshow(img)
plt.show()

##7. apply the threshold function onto the hsv image
ret,thresh2= cv2.threshold(hsv,127,255,cv2.THRESH_BINARY)
## thresh2 is our thresholded image
## show the thresholded image
plt.imshow(thresh2)
plt.show()


## convert the image into grayscale format
rgbimg = cv2.cvtColor(thresh2, cv2.COLOR_HSV2RGB)
theimg = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2GRAY)
## Now use findContours function to find all contours present in the ## image
## we will ignore first and last variables since only contours 
## variable will only be useful to our task.
contours, _ = cv2.findContours(theimg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours) )      ##  275
## Find the biggest contour among the list of contours
## use max function to find the maximum value with respect to
## contour area
biggest_contour = max(contours, key = cv2.contourArea)

## Create a copy of the original image to draw the contour
image_with_contour = image.copy()
## Draw the biggest contour on the image
cv2.drawContours(image_with_contour, [biggest_contour], -1, (0, 255, 0), 3)
## Convert the image to RGB for displaying with matplotlib
image_with_contour_rgb = cv2.cvtColor(image_with_contour, cv2.COLOR_BGR2RGB)
## Plot the image with the contour
plt.imshow(image_with_contour_rgb)
plt.show()