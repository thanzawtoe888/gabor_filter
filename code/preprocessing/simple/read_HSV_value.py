import cv2
import numpy as np

def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)  # [1] means width, [0] means height
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Load and resize the image
img = cv2.imread("../../images/frame_0015.jpg")
resized_img = rescale_frame(img)

rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

# Convert BGR to HSV
hsv_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
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
