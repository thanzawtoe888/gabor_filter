import cv2 as cv
import numpy as np

blank_image = np.zeros((500, 500), dtype= 'uint8')
cv.imshow('Blank Image', blank_image)


cv.waitKey(0)
cv.destroyAllWindows()
