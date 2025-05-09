import cv2 as cv
import numpy as np

blank = np.zeros((500, 500,3), dtype='uint8')


top_left1 = (0, 50)
bottom_right1 = (500, 90)

top_left = (0, 150)
bottom_right = (500, 300)

top_left2 = (0,400 )
bottom_right2 = (500, 450)

# Draw the rectangle
cv.rectangle(blank, top_left, bottom_right, (0, 255, 0), thickness=cv.FILLED)

# Draw the rectangle1
cv.rectangle(blank, top_left1, bottom_right1, (0, 0, 255), thickness=cv.FILLED)

# Draw the rectangle2
cv.rectangle(blank, top_left2, bottom_right2, (255, 0, 0), thickness=cv.FILLED)

# Show image
cv.imshow('Rectangle with Dimensions', blank)


cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite("../../results/three_rect.jpg", blank)  # Save the image with detected lines
