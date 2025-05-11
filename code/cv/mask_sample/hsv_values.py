import cv2 as cv
import numpy as np
rosy = np.uint8([[[245,220,217 ]]])
hsv_rosy = cv.cvtColor(rosy,cv.COLOR_BGR2HSV)
print( hsv_rosy )
