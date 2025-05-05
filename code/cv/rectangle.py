import cv2 as cv
import numpy as np

blank = np.zeros((500, 500,3), dtype='uint8')

top_left = (250, 250)
bottom_right = (500, 500)

# Draw the rectangle
cv.rectangle(blank, top_left, bottom_right, (0, 255, 0), thickness=2)

# Calculate width and height
width = bottom_right[0] - top_left[0]
height = bottom_right[1] - top_left[1]

# Put text for width
cv.putText(blank, f'Width: {width}px', (10, 270), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# Put text for height
cv.putText(blank, f'Height: {height}px', (260, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# Show image
cv.imshow('Rectangle with Dimensions', blank)


cv.waitKey(0)
cv.destroyAllWindows()
