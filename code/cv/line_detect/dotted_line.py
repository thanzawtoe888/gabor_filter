import cv2 as cv
import numpy as np

# Create a blank 500x500 image
image = np.zeros((500, 500, 3), dtype='uint8')

# Define dotted line parameters
start_x = 0
end_x = 500
y = 250  # Middle of the image
dot_length = 5
space_length = 5

# Draw the dotted line
for x in range(start_x, end_x, dot_length + space_length):
    cv.line(image, (x, y), (x + dot_length, y), (255, 255, 255), thickness=1)

# Show the image
cv.imshow('Dotted Line', image)
cv.waitKey(0)
cv.destroyAllWindows()
