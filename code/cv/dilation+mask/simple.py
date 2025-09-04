import cv2
import numpy as np

# Create a pink canvas (e.g., RGB(255, 182, 193))
height, width = 400, 600
pink_color = (193, 182, 255)  # OpenCV uses BGR
canvas = np.full((height, width, 3), pink_color, dtype=np.uint8)

# Define line endpoints
start_point = (100, 100)
end_point = (500, 300)
line_color = (0, 0, 0)  # Black
line_thickness = 1

# Draw the line on the pink canvas
before = canvas.copy()
cv2.line(before, start_point, end_point, line_color, line_thickness)

# Copy for after processing
after = before.copy()

# Now find pixels that are black (line), and fill surrounding pixels with nearest pink
mask = cv2.inRange(after, (0, 0, 0), (0, 0, 0))  # Find the black line
dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)  # Expand the line
line_outline = cv2.bitwise_xor(dilated, mask)  # Get surrounding area

# Fill the outline with pink color
after[line_outline == 255] = pink_color

# Concatenate images for side-by-side display
comparison = np.hstack((before, after))

# Show the result
cv2.imshow("Before (left) and After (right)", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()