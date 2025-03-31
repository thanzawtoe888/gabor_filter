import cv2
import numpy as np

# # Load image
# image = cv2.imread("../../images/frame_0013.jpg")
# import cv2
# import numpy as np

def process_image(image_path):
    # Load the image
    image = cv2.imread("../../images/frame_0013.jpg")
    if image is None:
        print("Error: Image not found!")
        return

    # Define the ROI coordinates [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    points = [(0.6, 775), (3833, 775), (0.6, 1496), (3833, 1496)]
    
    # Convert to integers (OpenCV needs whole pixels)
    points = [(int(x), int(y)) for x, y in points]
    
    # Get the bounding rectangle of the ROI
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)

    # 1. Crop the ROI
    roi = image[y1:y2, x1:x2]
    
    # 2. Resize the ROI (optional - to 800x600 in this example)
    resized_roi = cv2.resize(roi, (800, 600))
    
    # 3. Create a binary mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Black background
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)  # White rectangle for ROI
    
    # 4. Apply the mask to original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Save results
    cv2.imwrite("cropped_roi.jpg", roi)
    cv2.imwrite("resized_roi.jpg", resized_roi)
    cv2.imwrite("binary_mask.jpg", mask)
    cv2.imwrite("masked_image.jpg", masked_image)
    
    # Display results (optional)
    cv2.imshow("Original Image", image)
    cv2.imshow("Cropped ROI", roi)
    cv2.imshow("Resized ROI", resized_roi)
    cv2.imshow("Binary Mask", mask)
    cv2.imshow("Masked Image", masked_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the program
process_image("input.jpg")