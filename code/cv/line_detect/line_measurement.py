import cv2 as cv
import numpy as np

# Create a blank image
image = np.zeros((500, 500, 3), dtype="uint8")

# Draw a rectangle
# cv.rectangle(image, (50, 100), (300, 400), (0, 255, 0), thickness=2)  # Random position

# Draw a horizontal line
cv.line(image, (60, 90), (90, 250), (255, 0, 0), thickness=5, lineType= 16 )  # Blue line

# Convert to grayscale and find contours
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
contour,_ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# Iterate over contours
for contour in contour:
    x, y, w, h = cv.boundingRect(contour)  # Get the bounding box
    cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
    
    # Put width and height as text
    cv.putText(image, f'Width: {w}px', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(image, f'Height: {h}px', (x, y + h + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(image, f'Length: {w}px', (x, y + h + 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# Display the image
cv.imshow("Lines", image)


cv.waitKey(0)
cv.destroyAllWindows()