import cv2 as cv2
import numpy as np

import os


# Replace with the actual path to your image file in Google Drive
image_path = r'C:\Users\luca\Desktop\frames_720_final\frame_0000.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found at the specified path.")
else:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to get a binary image
    # Adjust the block size and C value as needed
    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -2)

    # Create horizontal and vertical kernels for morphological operations
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    # Apply morphological operations to detect horizontal lines
    detected_horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    horizontal_coords = np.argwhere(detected_horizontal_lines > 0)

    # Apply morphological operations to detect vertical lines
    detected_vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    vertical_coords = np.argwhere(detected_vertical_lines > 0)

    # Draw the detected horizontal and vertical lines on the original image
    output_image = image.copy()
    for coord in horizontal_coords:
        cv2.circle(output_image, (coord[1], coord[0]), 1, (0, 255, 0), -1) # Green color for horizontal

    for coord in vertical_coords:
        cv2.circle(output_image, (coord[1], coord[0]), 1, (255, 0, 0), -1) # Blue color for vertical


    # Display the original and processed images
    print("Original Image:")
    cv2.imshow("Original Image", image)

    print("\nBinary Image (after adaptive thresholding):")
    cv2.imshow("Binary Image (after adaptive thresholding)", binary)

    print("\nDetected Horizontal Lines:")
    cv2.imshow("Detected Horizontal Lines", detected_horizontal_lines)

    print("\nDetected Vertical Lines:")
    cv2.imshow("Detected Vertical Lines", detected_vertical_lines)

    print("\nOutput Image with Detected Lines Marked:")
    cv2.imshow("Output Image with Detected Lines Marked", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()