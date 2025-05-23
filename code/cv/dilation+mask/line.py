import cv2 as cv
import numpy as np
blank = np.zeros((500, 500,3), dtype='uint8')

pink_line = cv.line(blank, (0, 0), (400, 250), (255, 182, 193), thickness=2)
cv.imshow('Pink Line', pink_line)

mask = cv.inRange(pink_line, (255, 182, 193), (255, 182, 193))
cv.imshow('Mask', mask)
dilated = cv.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
line_outline = cv.bitwise_xor(dilated, mask)  # Get surrounding area
after = pink_line.copy()
after[line_outline == 255] = (255, 182, 193)  # Fill the outline with pink color
comparison = np.hstack((pink_line, after))
cv.imshow("Before (left) and After (right)", comparison)

cv.waitKey(0)

cv.destroyAllWindows()