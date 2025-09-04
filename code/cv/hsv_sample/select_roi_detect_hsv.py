import cv2 as cv
import numpy as np

# Global variables
drawing = False
ix, iy = -1, -1
rect = (0, 0, 0, 0)

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = blank.copy()
            cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv.imshow('Select Area', img_copy)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        rect = (ix, iy, x, y)
        cv.rectangle(blank, (ix, iy), (x, y), (0, 255, 0), 2)
        cv.imshow('Select Area', blank)

        # Extract the ROI and display HSV range
        x1, y1, x2, y2 = rect
        roi = blank[y1:y2, x1:x2]
        hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

        # Calculate min and max HSV values in the selected area
        min_hsv = np.min(hsv_roi.reshape(-1, 3), axis=0)
        max_hsv = np.max(hsv_roi.reshape(-1, 3), axis=0)

        print(f"HSV Range in Selected Area:")
        print(f"Min HSV: {min_hsv}")
        print(f"Max HSV: {max_hsv}")

# Create a blank image
blank = np.zeros((500, 500, 3), dtype='uint8')

# Draw sample rectangles (Red and Blue)
cv.rectangle(blank, (100, 100), (200, 300), (255, 0, 0), cv.FILLED)  # Blue rectangle
cv.rectangle(blank, (300, 150), (400, 350), (0, 0, 255), cv.FILLED)  # Red rectangle

# Set up the window and bind the function
cv.namedWindow('Select Area')
cv.setMouseCallback('Select Area', draw_rectangle)

# Display the image and wait for the user to select the area
while True:
    cv.imshow('Select Area', blank)
    key = cv.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        break

cv.destroyAllWindows()
