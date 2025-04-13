import cv2

# Load the original image
image = cv2.imread('../../images/frame_0013.jpg')
clone = image.copy()
points = []

# Resize only for display
display_image = cv2.resize(image, (1280, 720))  # Resize for easier viewing
scale_x = image.shape[1] / 1280  # width scaling factor
scale_y = image.shape[0] / 720   # height scaling factor

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Scale click points back to original size
        orig_x = int(x * scale_x)
        orig_y = int(y * scale_y)
        points.append((orig_x, orig_y))

        # Draw circle on display image (scaled)
        cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Image', display_image)

        # If 2 points selected, draw on original
        if len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            cropped = clone[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            cv2.imshow("Cropped", cropped)

# Show resized image
cv2.imshow('Image', display_image)
cv2.setMouseCallback('Image', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()
