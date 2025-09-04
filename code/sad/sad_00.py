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

def highlight_cracks(original_image_path, crack_map, output_path):
    """Overlay detected cracks on the original image using color map."""
    original = cv2.imread(original_image_path)
    crack_color = cv2.applyColorMap(crack_map, cv2.COLORMAP_JET)
    
    # Resize crack map to match original in case of slight shape difference
    crack_color = cv2.resize(crack_color, (original.shape[1], original.shape[0]))
    
    overlay = cv2.addWeighted(original, 0.7, crack_color, 0.3, 0)
    cv2.imwrite(output_path, overlay)
    print(f"Crack overlay saved to {output_path}")

# Paths to input images
reference_path = './images/frame_0000.jpg'
current_path = './images/frame_0015.jpg'

# Step 1: Preprocess both images
ref_preprocessed = preprocess_image(reference_path)
cur_preprocessed = preprocess_image(current_path)

# Step 2: Compute SAD
window_size = 5
sad_map = compute_sad(cur_preprocessed, ref_preprocessed, window_size)

# Step 3: Threshold the SAD map
threshold = 1000  # Adjust based on test
crack_map = detect_crack_regions(sad_map, threshold)

# Step 4: Save raw and colorized crack map
cv2.imwrite('./results/crack_map_raw.jpg', crack_map)
highlight_cracks(current_path, crack_map, './results/crack_overlay.jpg')
