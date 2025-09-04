import cv2 as cv
import numpy as np

def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale) ## [1] mean width, [0] mean height
    height = int(frame.shape[0] * scale) ## [1] mean width, [0] mean height
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

img = cv.imread("../images/frame_0013.jpg")
resized_img = rescale_frame(img)
cv.imshow("Resized Image", resized_img)

cv.imwrite("../results/rescale.jpg", resized_img)  # Save the resized image
#show the image resolution
print("Original Dimensions : ", img.shape)
print("Resized Dimensions : ", resized_img.shape)
cv.waitKey(0)
cv.destroyAllWindows()


print(np.shape(resized_img))