import cv2

# Load the image
image = cv2.imread('../../images/frame_0013.jpg')

# Crop the image: [y1:y2, x1:x2]
cropped_image = image[0:745, 0:3840]

# Show the cropped image
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
