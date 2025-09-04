import cv2
import numpy as np

def preprocess_image(image_path):
    """Load, grayscale, blur, and enhance contrast of an image."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    return enhanced

def compute_sad(current_image, reference_image, window_size):
    """Compute the SAD map between two preprocessed images."""
    if current_image.shape != reference_image.shape:
        raise ValueError("Images must have the same dimensions.")

    h, w = current_image.shape
    sad_map = np.zeros((h - window_size + 1, w - window_size + 1), dtype=np.float32)

    for y in range(h - window_size + 1):
        for x in range(w - window_size + 1):
            patch_cur = current_image[y:y+window_size, x:x+window_size]
            patch_ref = reference_image[y:y+window_size, x:x+window_size]
            sad = np.sum(np.abs(patch_cur - patch_ref))
            sad_map[y, x] = sad

    return sad_map

def detect_crack_regions(sad_map, threshold):
    """Threshold the SAD map to find potential crack regions."""
    crack_map = (sad_map > threshold).astype(np.uint8) * 255
    return crack_map

def highlight_cracks(original_image_path, crack_map, bbox_coords, output_path):
    """Overlay detected cracks on the original image using a color map and bounding box."""
    original = cv2.imread(original_image_path)
    
    # Create a blank image same size as original
    full_crack_overlay = np.zeros_like(original)

    # Resize crack_map to fit the bounding box size
    x1, y1 = bbox_coords[0]
    x2, y2 = bbox_coords[2]
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    resized_crack = cv2.resize(crack_map, (bbox_width, bbox_height))

    # Apply color map to the crack detection
    color_map = cv2.applyColorMap(resized_crack, cv2.COLORMAP_JET)
    
    # Overlay only inside the bounding box
    full_crack_overlay[y1:y2, x1:x2] = color_map
    overlay = cv2.addWeighted(original, 0.7, full_crack_overlay, 0.3, 0)

    # Draw the bounding box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save result
    cv2.imwrite(output_path, overlay)
    print(f"Crack overlay with bounding box saved to {output_path}")

# ---------- Main Execution ---------- #

# Define bounding box (top-left and bottom-right points)
points = [(0, 775), (3833, 775), (3833, 1496), (0, 1496)]
bbox_coords = [points[0], points[1], points[2], points[3]]  # (x1, y1), (x2, y2)

# Paths
reference_path = './images/frame_0000.jpg'
current_path = './images/frame_0015.jpg'

# Load and preprocess
ref_full = preprocess_image(reference_path)
cur_full = preprocess_image(current_path)

# Crop to bounding box
x1, y1 = bbox_coords[0]
x2, y2 = bbox_coords[2]
ref_crop = ref_full[y1:y2, x1:x2]
cur_crop = cur_full[y1:y2, x1:x2]

# Compute SAD
window_size = 5
sad_map = compute_sad(cur_crop, ref_crop, window_size)

# Threshold
threshold = 1000
crack_map = detect_crack_regions(sad_map, threshold)

# Save outputs
cv2.imwrite('./results/crack_map_bbox_raw.jpg', crack_map)
highlight_cracks(current_path, crack_map, bbox_coords, './results/crack_overlay_bbox.jpg')
