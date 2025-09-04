import cv2
import numpy as np

def preprocess_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(blurred_image)

    # Save the preprocessed image
    cv2.imwrite(output_path, enhanced_image)

    def compute_sad(current_image_path, reference_image_path, window_size):
        # Load the current and reference images
        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

        # Ensure both images have the same dimensions
        if current_image.shape != reference_image.shape:
            raise ValueError("Current and reference images must have the same dimensions.")

        # Get image dimensions
        height, width = current_image.shape

        # Initialize SAD map
        sad_map = np.zeros((height - window_size + 1, width - window_size + 1))

        # Sliding window to compute SAD
        for y in range(0, height - window_size + 1):
            for x in range(0, width - window_size + 1):
                # Extract patches
                current_patch = current_image[y:y + window_size, x:x + window_size]
                reference_patch = reference_image[y:y + window_size, x:x + window_size]

                # Compute SAD for the current window
                sad = np.sum(np.abs(current_patch - reference_patch))
                sad_map[y, x] = sad

        return sad_map

    # Example usage
    reference_image_path = 'reference_image.jpg'  # Replace with your reference image path
    window_size = 5  # Define the size of the sliding window
    sad_map = compute_sad(output_image_path, reference_image_path, window_size)
    print("SAD map computed.")

    # Define a threshold for SAD values
    threshold = 1000  # Adjust this value based on your application

    # Apply thresholding to detect regions of interest
    crack_map = (sad_map > threshold).astype(np.uint8) * 255

    # Save the crack map as an image
    crack_map_output_path = 'crack_map.jpg'  # Replace with your desired output path
    cv2.imwrite(crack_map_output_path, crack_map)

    print(f"Crack map saved to {crack_map_output_path}.")
    # Highlight regions with high SAD values
    highlighted_crack_map = cv2.applyColorMap(crack_map, cv2.COLORMAP_JET)

    # Save the highlighted crack map
    highlighted_crack_map_output_path = 'highlighted_crack_map.jpg'  # Replace with your desired output path
    cv2.imwrite(highlighted_crack_map_output_path, highlighted_crack_map)
    print(f"Highlighted crack map saved to {highlighted_crack_map_output_path}.")

    # Use morphological operations to trace continuous crack lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    traced_crack_map = cv2.morphologyEx(crack_map, cv2.MORPH_CLOSE, kernel)

    # Save the traced crack map
    traced_crack_map_output_path = 'traced_crack_map.jpg'  # Replace with your desired output path
    cv2.imwrite(traced_crack_map_output_path, traced_crack_map)
    print(f"Traced crack map saved to {traced_crack_map_output_path}.")

    # Map detected cracks over time for propagation analysis
    # (This part assumes you have multiple crack maps over time to analyze)
    # Example: Load multiple crack maps and compute differences
    # crack_map_t1 = cv2.imread('crack_map_t1.jpg', cv2.IMREAD_GRAYSCALE)
    # crack_map_t2 = cv2.imread('crack_map_t2.jpg', cv2.IMREAD_GRAYSCALE)
    # crack_propagation = cv2.absdiff(crack_map_t2, crack_map_t1)

    # Save the crack propagation map
    # crack_propagation_output_path = 'crack_propagation.jpg'
    # cv2.imwrite(crack_propagation_output_path, crack_propagation)
    # print(f"Crack propagation map saved to {crack_propagation_output_path}.")

# Function to create a frame without a crack
def create_no_crack_frame(output_path):
    frame = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White frame
    cv2.imwrite(output_path, frame)
    print(f"No-crack frame saved to {output_path}.")

# Function to create a frame with a crack
def create_crack_frame(output_path):
    frame = np.ones((200, 200, 3), dtype=np.uint8) * 255  # White frame
    cv2.line(frame, (50, 50), (150, 150), (0, 0, 0), 2)  # Draw a black crack line
    cv2.imwrite(output_path, frame)
    print(f"Crack frame saved to {output_path}.")

# Example usage
no_crack_frame_path = 'no_crack_frame.jpg'
crack_frame_path = 'crack_frame.jpg'
create_no_crack_frame(no_crack_frame_path)
create_crack_frame(crack_frame_path)

input_image_path = './images/frame_0000.jpg'  # Replace with your input image path
output_image_path = './images/frame_0015.jpg'  # Replace with your desired output path
preprocess_image(input_image_path, output_image_path)