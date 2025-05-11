import cv2 as cv
import numpy as np

# Global variables
ref_point = []
cropping = False

def set_hsv_range(hue, saturation_percentage, value_percentage):
    """
    Convert HSV values to OpenCV scale.

    Args:
        hue (int): Hue value (0-360 degrees).
        saturation_percentage (int): Saturation as a percentage (0-100).
        value_percentage (int): Value as a percentage (0-100).

    Returns:
        np.ndarray: The HSV value in OpenCV format.
    """
    # Convert saturation and value from percentage to 0-255 scale
    saturation = int((saturation_percentage / 100) * 255)
    value = int((value_percentage / 100) * 255)

    # OpenCV Hue range is 0-179 (not 0-360), so we adjust
    hue = int((hue / 360) * 179)

    return np.array([hue, saturation, value])

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping

    # Start cropping
    if event == cv.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # Update rectangle during dragging
    elif event == cv.EVENT_MOUSEMOVE:
        if cropping:
            image_copy = image.copy()
            cv.rectangle(image_copy, ref_point[0], (x, y), (0, 255, 0), 2)
            cv.imshow("Select Rectangle", image_copy)

    # Complete cropping
    elif event == cv.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # Draw the final rectangle
        cv.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv.imshow("Select Rectangle", image)

def detect_hsv_range(cropped_image, lower_hsv, upper_hsv):
    """
    Detects areas within the specified HSV range and displays the mask.

    Args:
        cropped_image (np.ndarray): The cropped region of the image.
        lower_hsv (np.ndarray): Lower HSV range.
        upper_hsv (np.ndarray): Upper HSV range.
    """
    # Apply Gaussian Blur
    blurred = cv.GaussianBlur(cropped_image, (5, 5), 0)

    # Convert to HSV
    hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

    # Create mask
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)

    # Apply mask to the image
    result = cv.bitwise_and(cropped_image, cropped_image, mask=mask)

    # Display the mask and the result
    cv.imshow("HSV Mask", mask)
    cv.imshow("Detected Range", result)

    # Calculate min and max HSV values in the cropped area
    hsv_pixels = hsv.reshape(-1, 3)
    min_hsv = np.min(hsv_pixels, axis=0)
    max_hsv = np.max(hsv_pixels, axis=0)

    print(f"Minimum HSV: {min_hsv}")
    print(f"Maximum HSV: {max_hsv}")

def main(resized_image):

    # Input HSV Range in user-friendly format
    print("Enter lower HSV range (Hue (0-360), Saturation (0-100), Value (0-100)): ")
    lower_hue = int(input("Lower Hue: "))
    lower_sat = int(input("Lower Saturation: "))
    lower_val = int(input("Lower Value: "))

    print("Enter upper HSV range (Hue (0-360), Saturation (0-100), Value (0-100)): ")
    upper_hue = int(input("Upper Hue: "))
    upper_sat = int(input("Upper Saturation: "))
    upper_val = int(input("Upper Value: "))

    # Convert user input to OpenCV HSV range
    lower_hsv = set_hsv_range(lower_hue, lower_sat, lower_val)
    upper_hsv = set_hsv_range(upper_hue, upper_sat, upper_val)

    print(f"Lower HSV: {lower_hsv}")
    print(f"Upper HSV: {upper_hsv}")

    cv.imshow("Select Rectangle", resized_image)
    cv.setMouseCallback("Select Rectangle", click_and_crop)

    while True:
        key = cv.waitKey(1) & 0xFF

        # Press 'c' to crop the selected area and detect HSV range
        if key == ord("c") and len(ref_point) == 2:
            x1, y1 = ref_point[0]
            x2, y2 = ref_point[1]

            # Ensure coordinates are in the correct order
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Crop the selected area
            cropped_image = image[y1:y2, x1:x2]

            # Display and save the cropped image
            cv.imshow("Cropped Image", cropped_image)
            cv.imwrite("cropped_image.jpg", cropped_image)

            # Detect HSV range in the cropped area
            detect_hsv_range(cropped_image, lower_hsv, upper_hsv)

            cv.waitKey(0)

        # Press 'q' to exit
        elif key == ord("q"):
            break

    cv.destroyAllWindows()

# Example usage
def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] // 5)  # [1] means width, [0] means height
    height = int(frame.shape[0] // 5)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Load and resize the image
img = cv.imread("../../images/three_rect.jpg")
resized_img = rescale_frame(img)

main(resized_img)
