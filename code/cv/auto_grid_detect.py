import cv2 as cv
import numpy as np
import math

def analyze_grid_lines(image_path):
    """
    Analyzes an image to detect grid lines, categorize them, find their
    intersections, and visualize the results.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        None: The function displays the image with detected lines and
              intersections. It also prints the found intersection points.
    """
    # 1. Image Loading and Preprocessing
    try:
        image = cv.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from path: {image_path}")
            return
    except Exception as e:
        print(f"An error occurred while reading the image: {e}")
        return

    # Create a copy for drawing on later
    output_image = image.copy()

    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred_image = cv.GaussianBlur(gray_image, (5, 5), 0)

    # Use Canny edge detection
    edges = cv.Canny(blurred_image, 50, 150, apertureSize=3)

    # 2. Line Detection using Probabilistic Hough Line Transform
    #    rho: The resolution of the parameter r in pixels. We use 1 pixel.
    #    theta: The resolution of the parameter theta in radians. We use 1 degree (np.pi/180).
    #    threshold: The minimum number of votes (intersections in Hough grid)
    #    minLineLength: The minimum number of points that can form a line.
    #    maxLineGap: The maximum gap in pixels between connectable line segments.
    lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        print("No lines were detected in the image.")
        return

    # 3. Process and Calculate
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Draw the raw detected lines in blue
        cv.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Calculate the angle of the line
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # Categorize lines into horizontal and vertical
        # We allow a small tolerance (e.g., +/- 5 degrees)
        if abs(angle) < 5 or abs(angle - 180) < 5 or abs(angle + 180) < 5:
            horizontal_lines.append(line[0])
        elif abs(angle - 90) < 5 or abs(angle + 90) < 5:
            vertical_lines.append(line[0])

    print(f"Detected {len(horizontal_lines)} horizontal lines.")
    print(f"Detected {len(vertical_lines)} vertical lines.")

    # Calculate intersection points
    intersection_points = []
    for h_line in horizontal_lines:
        x1h, y1h, x2h, y2h = h_line
        for v_line in vertical_lines:
            x1v, y1v, x2v, y2v = v_line

            # Line equations: Ax + By = C
            # Horizontal line
            Ah = y2h - y1h
            Bh = x1h - x2h
            Ch = Ah * x1h + Bh * y1h
            # Vertical line
            Av = y2v - y1v
            Bv = x1v - x2v
            Cv = Av * x1v + Bv * y1v

            determinant = Ah * Bv - Av * Bh

            if determinant != 0:
                # Cramer's rule to find intersection
                intersect_x = (Bv * Ch - Bh * Cv) / determinant
                intersect_y = (Ah * Cv - Av * Ch) / determinant
                intersection_points.append((int(intersect_x), int(intersect_y)))


    print(f"\nFound {len(intersection_points)} intersection points:")
    for point in intersection_points:
        print(point)
        # Draw intersection points as red circles on the output image
        cv.circle(output_image, point, 5, (0, 0, 255), -1)


    # Display the final image
    cv.imshow('Detected Grid Lines and Intersections', output_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def create_sample_grid_image(filename="sample_grid.png", width=500, height=500):
    """Creates and saves a simple grid image for testing."""
    # Create a white background
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    color = (0, 0, 0) # Black lines
    thickness = 2

    # Draw vertical lines
    for x in range(50, width, 100):
        cv.line(image, (x, 0), (x, height), color, thickness)

    # Draw horizontal lines
    for y in range(50, height, 100):
        cv.line(image, (0, y), (width, y), color, thickness)

    cv.imwrite(filename, image)
    print(f"Sample grid image saved as '{filename}'")
    return filename

if __name__ == '__main__':
    # Create a sample image to run the analysis on
    sample_image_path = create_sample_grid_image()

    # Run the analysis function
    analyze_grid_lines(sample_image_path)
