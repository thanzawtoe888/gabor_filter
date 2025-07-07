# import cv2 as cv
# import numpy as np


# resized_img = cv.imread("../results/drawed_gridline.jpg")

# crack_shape = np.shape(resized_img)
# print("Crack Image Shape:",crack_shape)
# # cv.imshow("Resized Image", resized_img)

# # Convert to grayscale and apply Gaussian blur
# gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
# blurred = cv.GaussianBlur(gray, (5, 5), 0)

# # Apply Canny edge detection
# edges = cv.Canny(blurred, threshold1=50, threshold2=150)

# # Apply Hough Line Transform
# lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# # Draw the detected lines on the image
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv.line(resized_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)  # Green lines for detected edges

# # Display the image with detected lines
# cv.imshow ("Canny Edges", edges)
# cv.imshow('Measured Rectangle', resized_img)

# # cv.imwrite("../results/detected_lines_.jpg", edges)  # Save the image with detected lines
# cv.waitKey(0)
# cv.destroyAllWindows()

import cv2
import numpy as np

def detect_cracks(image_path, canny_threshold1=50, canny_threshold2=150, hough_threshold=100, min_line_length=50, max_line_gap=10):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detector
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)

    # Use Hough Line Transform to detect straight lines (potential cracks)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Draw detected lines on the original image
    output = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red lines for cracks

    # Display the result
    cv2.imshow("Original Image", image)
    cv2.imshow("Canny Edges", edges)
    cv2.imshow("Detected Cracks", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'road.jpg' with the path to your road image
    detect_cracks("../images/frame_0015.jpg")