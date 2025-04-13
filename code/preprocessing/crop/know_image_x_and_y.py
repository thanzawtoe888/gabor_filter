import cv2

# Load the image
image = cv2.imread('../../images/frame_0015.jpg')

# Get height, width, and channels
height, width, channels = image.shape

print("Height (y):", height)
print("Width (x):", width)
print("Channels:", channels)

#show the image
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
