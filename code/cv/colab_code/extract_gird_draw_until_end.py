import cv2
import numpy as np
import os

# Replace with the actual path to your image file
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

    # Create a copy of the original image to draw lines on
    output_image = image.copy()
    height, width, _ = image.shape

    # --- Process and Draw Horizontal Lines ---
    if len(horizontal_coords) > 0:
        # Sort horizontal coordinates by row
        horizontal_coords = horizontal_coords[horizontal_coords[:, 0].argsort()]
        
        current_line_rows = []
        grouped_horizontal_lines = []

        for r, c in horizontal_coords:
            if not current_line_rows:
                current_line_rows.append(r)
            elif abs(r - current_line_rows[-1]) < 5:  # Group rows within 5 pixels
                current_line_rows.append(r)
            else:
                grouped_horizontal_lines.append(int(np.mean(current_line_rows)))
                current_line_rows = [r]
        if current_line_rows:
            grouped_horizontal_lines.append(int(np.mean(current_line_rows)))

        for row_pos in grouped_horizontal_lines:
            cv2.line(output_image, (0, row_pos), (width - 1, row_pos), (0, 255, 0), 2) # Green line, thickness 2

    # --- Process and Draw Vertical Lines ---
    if len(vertical_coords) > 0:
        # Sort vertical coordinates by column
        vertical_coords = vertical_coords[vertical_coords[:, 1].argsort()]

        current_line_cols = []
        grouped_vertical_lines = []

        for r, c in vertical_coords:
            if not current_line_cols:
                current_line_cols.append(c)
            elif abs(c - current_line_cols[-1]) < 5:  # Group columns within 5 pixels
                current_line_cols.append(c)
            else:
                grouped_vertical_lines.append(int(np.mean(current_line_cols)))
                current_line_cols = [c]
        if current_line_cols:
            grouped_vertical_lines.append(int(np.mean(current_line_cols)))

        for col_pos in grouped_vertical_lines:
            cv2.line(output_image, (col_pos, 0), (col_pos, height - 1), (255, 0, 0), 2) # Blue line, thickness 2

    # Display the images
    print("Original Image:")
    cv2.imshow("Original Image", image)

    print("\nBinary Image (after adaptive thresholding):")
    cv2.imshow("Binary Image (after adaptive thresholding)", binary)

    print("\nDetected Horizontal Lines (Binary):")
    cv2.imshow("Detected Horizontal Lines (Binary)", detected_horizontal_lines)

    print("\nDetected Vertical Lines (Binary):")
    cv2.imshow("Detected Vertical Lines (Binary)", detected_vertical_lines)

    print("\nOutput Image with Detected Grid Lines:")
    cv2.imshow("Output Image with Detected Grid Lines", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()