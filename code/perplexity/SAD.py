import cv2
import numpy as np

def resize_and_convert_to_intensity(image_path, size=(100, 100)):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    
    # Resize the image
    resized = cv2.resize(image, size)

    # Convert to grayscale (intensity)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    return gray

def compute_sad(image1, image2):
    # Compute SAD (Sum of Absolute Differences)
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same size for SAD comparison.")

    sad = np.sum(np.abs(image1.astype(np.int16) - image2.astype(np.int16)))
    return sad

# Example usage
image1_path = 'image1.jpg'
image2_path = 'image2.jpg'

gray1 = resize_and_convert_to_intensity(image1_path)
gray2 = resize_and_convert_to_intensity(image2_path)

sad_value = compute_sad(gray1, gray2)
print(f"Sum of Absolute Differences (SAD): {sad_value}")
