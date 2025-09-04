import cv2 as cv
import numpy as np

blank = np.zeros((500, 500,3), dtype='uint8')

top_left = (150, 150)
bottom_right = (250, 250)

top_left_1 = (300, 300)
bottom_right_2 = (450, 450)
# Draw the rectangle
cv.rectangle(blank, top_left, bottom_right, (0, 255, 0), cv.FILLED )

# Draw the rectangle with a different color
cv.rectangle(blank, top_left_1, bottom_right_2, (255, 0, 0), cv.FILLED )

# use hsv to find the color
hsv = cv.cvtColor(blank, cv.COLOR_BGR2HSV)

# show the hsv image
cv.imshow('HSV Image', hsv)
cv.imwrite('../results/todel_image.jpg', hsv)
# Define the color range for the rectangle
lower_color = np.array([30, 196, 255])
upper_color = np.array([60, 255, 255])
# Create a mask for the color
mask = cv.inRange(hsv, lower_color, upper_color)
# Apply the mask to the image
masked_image = cv.bitwise_and(blank, blank, mask=mask)
# Show the masked image
cv.imshow('Masked Image', masked_image)

# # Calculate width and height
# width = bottom_right[0] - top_left[0]
# height = bottom_right[1] - top_left[1]

# # Put text for width
# cv.putText(blank, f'Width: {width}px', (10, 270), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# # Put text for height
# cv.putText(blank, f'Height: {height}px', (260, 130), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# Show image
cv.imshow('Rectangle with Dimensions', blank)

#find the edge of the rectangle
edges = cv.Canny(masked_image, 100, 200)
cv.imshow('Edges', edges)
# Apply Hough Line Transform
lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
# Draw the detected lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(blank, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)  # Blue lines for detected edges
        # Draw the rectangle around the detected lines
        cv.rectangle(blank, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
        
# Display the results
cv.imshow("Detected Lines and Rectangle", blank)


cv.waitKey(0)
cv.destroyAllWindows()
