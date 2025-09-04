import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('../images/frame_0013.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges
lower_orange = np.array([5, 100, 100])
upper_orange = np.array([20, 255, 255])

lower_green = np.array([50, 100, 100])
upper_green = np.array([80, 255, 255])

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

# Orange device mask
mask_orange = cv2.inRange(image_hsv, lower_orange, upper_orange)

# Green strap mask
mask_green = cv2.inRange(image_hsv, lower_green, upper_green)

# Red padding mask (handle two ranges for red)
mask_red1 = cv2.inRange(image_hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(image_hsv, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Extract parts
orange_part = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_orange)
green_part = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_green)
red_part = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_red)

# Show each segmented part
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].imshow(orange_part)
axs[0].set_title('Orange Device')
axs[0].axis('off')

axs[1].imshow(green_part)
axs[1].set_title('Green Strap')
axs[1].axis('off')

axs[2].imshow(red_part)
axs[2].set_title('Red Padding')
axs[2].axis('off')

plt.tight_layout()
plt.show()

# Calculate average HSV value inside each mask
mean_hsv_orange = cv2.mean(image_hsv, mask=mask_orange)[:3]
mean_hsv_green = cv2.mean(image_hsv, mask=mask_green)[:3]
mean_hsv_red = cv2.mean(image_hsv, mask=mask_red)[:3]

print("Mean HSV for Orange Device:", mean_hsv_orange)
print("Mean HSV for Green Strap:", mean_hsv_green)
print("Mean HSV for Red Padding:", mean_hsv_red)
