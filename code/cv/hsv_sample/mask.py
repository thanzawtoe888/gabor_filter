import cv2 as cv
import numpy as np

def detect_hsv_range(image):
    """
    Detects areas within a specific HSV range and displays the mask.

    Args:
        image_path (str): Path to the input image.
    """
    # # Load the image
    # image = cv.imread(image_path)
    # if image is None:
    #     print("Error: Could not load image.")
    #     return

    # apply gaussian blur to the image
    image = cv.GaussianBlur(image, (5, 5), 0)
    # Convert to HSV color space
    bgr= cv.cvtColor(image, cv.COLOR_BGR2RGB)
    cv.imshow("BGR Image", bgr)
    rgb= cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    cv.imshow("BGR Image", rgb)
    hsv = cv.cvtColor(rgb, cv.COLOR_BGR2HSV)
    cv.imshow("HSV Image", hsv)

    # Define HSV range
    lower_hsv = np.array([0, 10, 80])
    upper_hsv = np.array([21, 15, 100])
  
    # Create mask
    mask = cv.inRange(hsv, lower_hsv, upper_hsv)
    

    # Apply the mask to the image
    result = cv.bitwise_and(image, image, mask=mask)

    # Display the mask and the result
    # cv.imshow("Original Image", image)
    # cv.imshow("HSV Mask", mask)
    # cv.imshow("Detected Range", result)
    cv.waitKey(0)
    cv.destroyAllWindows()

def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] // 5)  # [1] means width, [0] means height
    height = int(frame.shape[0] // 5)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Load and resize the image
img = cv.imread("../../images/frame_0013.jpg")
resized_img = rescale_frame(img)

# # Example usage
# image_path = "../../images/frame_0015.jpg"  # Replace with your image path

detect_hsv_range(resized_img)
