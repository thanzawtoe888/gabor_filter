import cv2
import numpy as np

# Load image
image = cv2.imread('../../images/frame_0015.jpg')

# Convert BGR to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image", hsv_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Example: select a 100x100 region from (x=50, y=50)
x, y, w, h = 50, 50, 100, 100
roi_hsv = hsv_image[y:y+h, x:x+w]
cv2.imshow("Region of Interest", roi_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get average HSV values in the region
mean_hsv = cv2.mean(roi_hsv)[:3]  # Ignore alpha channel if present

print("Average HSV in region:", mean_hsv)
