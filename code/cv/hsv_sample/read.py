import cv2 as cv
import numpy as np

def crop_green_rectangle(image_path):
    """
    Detects and crops the green rectangle from the image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Cropped image of the green rectangle, or None if no green rectangle is found.
    """
    # Load the image
    image = cv.imread(image_path)
    if image is None:
        print("Error: Could not load image.")
        return None

    # Convert to HSV to isolate the green color
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Define the green color range in HSV
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    # Create a mask for green
    mask = cv.inRange(hsv, lower_green, upper_green)

    # Find contours based on the mask
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Initialize a variable for the bounding rectangle
    green_rect = None

    # Iterate over contours to find the largest green rectangle
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        area = w * h
        # Assuming the largest detected green area is the desired rectangle
        if green_rect is None or area > green_rect[2] * green_rect[3]:
            green_rect = (x, y, w, h)

    # Crop the green rectangle
    if green_rect:
        x, y, w, h = green_rect
        cropped_green = image[y:y + h, x:x + w]
        return cropped_green
    else:
        print("No green rectangle detected.") 
        return None
    
    
        cv.waitKey(0)
        cv.destroyAllWindows()



cropped_image = crop_green_rectangle("../../results/three_rect.jpg")
cv.imshow("Cropped Green Rectangle", cropped_image)
if cropped_image is not None:
    cv.imwrite("../../results/cropped_green_rectangle.jpg", cropped_image)  # Save the cropped image
cv.waitKey(0)
cv.destroyAllWindows()

