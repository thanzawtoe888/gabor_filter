import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path, output_folder):
    # Load image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Create a copy of the original image to draw lines
    line_img = img.copy()

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Draw only vertical and horizontal lines
            if abs(x1 - x2) < 10:  # vertical line
                cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif abs(y1 - y2) < 10:  # horizontal line
                cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Save results
    cv2.imwrite(f'{output_folder}/grayscale.jpg', gray)
    cv2.imwrite(f'{output_folder}/edges.jpg', edges)
    cv2.imwrite(f'{output_folder}/detected_lines.jpg', line_img)

    # Display results
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title('Grayscale')
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Canny Edge Detection')
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Detected Lines')
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
process_image('../images/frame_0015.jpg', '../results')
