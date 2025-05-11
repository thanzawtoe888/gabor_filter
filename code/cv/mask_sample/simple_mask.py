import cv2 
import numpy as np
# Example usage
def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] // 5)  # [1] means width, [0] means height
    height = int(frame.shape[0] // 5)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

# Load and resize the image
img = cv2.imread("../../images/frame_0013.jpg")
image = rescale_frame(img)

print (image.shape)
# Check if the image was loaded successfully
if image is None:
    print("Error: Could not open or find the image.")
    exit()

# Create a blank mask with the same dimensions as the image
mask = np.zeros(image.shape[:2], dtype="uint8")

# Draw a filled rectangle on the mask
cv2.rectangle(mask, (0, 50), (500, 400), 255, -1)  # Rectangle 1

# Apply the mask using bitwise AND
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Convert the image to HSV format
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the HSV color range
target_hsv = np.array([117, 29, 245])
lower_bound = target_hsv - np.array([10, 10, 10])  # Adjust tolerance as needed
upper_bound = target_hsv + np.array([10, 10, 10])

# Create a mask for the specified HSV range
hsv_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

# Apply the HSV mask using bitwise AND
hsv_masked_image = cv2.bitwise_and(image, image, mask=hsv_mask)

# Display the original image, mask, and masked image
cv2.imshow("Original Image", image)
cv2.imshow("Mask", mask)
cv2.imshow("Masked Image", masked_image)

# Display the HSV mask and the masked image
cv2.imshow("HSV Mask", hsv_mask)
cv2.imshow("HSV Masked Image", hsv_masked_image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
