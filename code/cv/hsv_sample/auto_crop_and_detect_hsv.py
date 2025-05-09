import cv2 as cv
import numpy as np

# Global variables
ref_point = []
cropping = False

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

def detect_hsv_range(cropped_image):
    """
    Detects the minimum and maximum HSV values in the cropped image.

    Args:
        cropped_image (np.ndarray): The cropped region of the image.
    """
    # Convert to HSV
    hsv_image = cv.cvtColor(cropped_image, cv.COLOR_BGR2HSV)

    # Reshape to a list of pixels
    hsv_pixels = hsv_image.reshape(-1, 3)

    # Find min and max values
    min_hsv = np.min(hsv_pixels, axis=0)
    max_hsv = np.max(hsv_pixels, axis=0)

    print(f"Minimum HSV: {min_hsv}")
    print(f"Maximum HSV: {max_hsv}")

def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] // 5)  # [1] means width, [0] means height
    height = int(frame.shape[0] // 5)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def main(image_path):
    global image
    original_image = cv.imread(image_path)

    if original_image is None:
        print("Error: Could not load image.")
        return

    # Resize the image
    image = rescale_frame(original_image)

    cv.imshow("Select Rectangle", image)
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

            # Check if the cropped region is valid
            if cropped_image.size == 0:
                print("Error: Cropped region is empty. Please select a valid area.")
                continue

            # Display and save the cropped image
            cv.imshow("Cropped Image", cropped_image)
            cv.imwrite("cropped_image.jpg", cropped_image)

            # Detect HSV range
            detect_hsv_range(cropped_image)

            cv.waitKey(0)

        # Press 'q' to exit without cropping
        elif key == ord("q"):
            break

    cv.destroyAllWindows()

# Load and pass the image path to main
main("../../images/frame_0013.jpg")