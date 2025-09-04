import cv2

image = cv2.imread('../../images/frame_0013.jpg')

# Get original dimensions
original_height, original_width = image.shape[:2]

# Target height
target_height = 745

# Compute new width to preserve aspect ratio
aspect_ratio = original_width / original_height
new_width = int(target_height * aspect_ratio)

# Resize
resized_image = cv2.resize(image, (new_width, target_height))

cv2.imshow("Aspect Ratio Resized", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

