import cv2
import numpy as np

def detect_grid_lines(image, edges, 
                      rho=1, theta=np.pi / 180, 
                      threshold=80, 
                      min_line_length=80, 
                      max_line_gap=5):
    """
    Detects straight lines using Probabilistic Hough Transform.

    Parameters:
        image (np.array): Original RGB image to draw lines on.
        edges (np.array): Edge-detected binary image (from Canny).
        rho (float): Distance resolution in pixels.
        theta (float): Angle resolution in radians.
        threshold (int): Minimum number of intersections in Hough accumulator.
        min_line_length (int): Minimum length of line to be detected.
        max_line_gap (int): Maximum allowed gap between line segments.

    Returns:
        grid_line_img (np.array): Image with detected lines drawn.
        line_list (list): List of line coordinates [(x1, y1, x2, y2), ...]
    """
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, 
                            minLineLength=min_line_length, 
                            maxLineGap=max_line_gap)

    grid_line_img = image.copy()
    line_list = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(grid_line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            line_list.append((x1, y1, x2, y2))

    return grid_line_img, line_list


# Load and preprocess image
image = cv2.imread('../imgaes/frame_0015.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

# Call the Hough line function
grid_img, lines = detect_grid_lines(image, edges, threshold=80)

# Save output image
cv2.imwrite('../results/detected_grid_lines.jpg', grid_img)

# Optional: print line coordinates
print("Detected lines:", lines)
