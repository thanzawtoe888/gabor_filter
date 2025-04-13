import cv2

# Load the image
image = cv2.imread('../../images/frame_0013.jpg')
clone = image.copy()
points = []

# Mouse callback function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Save the point
        points.append((x, y))
        # Draw a small circle at the point
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', image)

        # If two points are selected, draw a rectangle
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cropped = clone[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
            cv2.imshow("Cropped", cropped)

# Show the image and set the mouse callback
cv2.imshow('Image', image)
cv2.setMouseCallback('Image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
