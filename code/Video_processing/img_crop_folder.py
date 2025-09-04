import cv2
import numpy as np
import os

# Set your input and output folder paths
input_path = r'C:\Users\luca\Desktop\frames_720_final'
output_path = r'C:\Users\luca\Desktop\frames_720_final\cropped_images'

# Create output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# List all image files in the input folder
for filename in os.listdir(input_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(input_path, filename)
        img = cv2.imread(image_path)

        if img is not None:
            # Optional: Show original shape
            print(f"Processing {filename}, shape: {img.shape}")

            # Crop region: [y1:y2, x1:x2]
            cropped_image = img[160:600, 0:1280]  # Adjust these values as needed

            # Save the cropped image
            cropped_filename = os.path.join(output_path, f"cropped_{filename}")
            cv2.imwrite(cropped_filename, cropped_image)
        else:
            print(f"Failed to load image: {filename}")

cv2.destroyAllWindows()
print("Cropping complete.")
