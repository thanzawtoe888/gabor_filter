import cv2 as cv
import numpy as np

def rescale_frame(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Load and resize the image
img = cv.imread("../results/drawed_gridline.jpg")
resized_img = rescale_frame(img)

# Create a mask with the same dimensions as the resized image
mask = np.zeros((resized_img.shape[0], resized_img.shape[1]), dtype=np.uint8)

# Define line thickness
thickness = 5

# Draw the lines (both on the image and the mask)
# Horizontal Blue Lines
cv.line(resized_img, (0, 10), (1920, 10), (255, 0, 0), thickness, lineType=16)
cv.line(mask, (0, 10), (1920, 10), 255, thickness, lineType=16)

cv.line(resized_img, (0, 125), (1920, 125), (255, 0, 0), thickness, lineType=16)
cv.line(mask, (0, 125), (1920, 125), 255, thickness, lineType=16)

cv.line(resized_img, (0, 240), (1920, 240), (255, 0, 0), thickness, lineType=16)
cv.line(mask, (0, 240), (1920, 240), 255, thickness, lineType=16)

cv.line(resized_img, (0, 366), (1920, 366), (255, 0, 0), thickness, lineType=16)
cv.line(mask, (0, 366), (1920, 366), 255, thickness, lineType=16)

# Vertical Green Lines
x_coords = [85, 260, 435, 615, 795, 970, 1145, 1325, 1500, 1675, 1848]

for x in x_coords:
    cv.line(resized_img, (x, 0), (x, 375), (0, 255, 0), thickness)
    cv.line(mask, (x, 0), (x, 375), 255, thickness)

# Apply inpainting to remove the drawn lines
inpainted_img = cv.inpaint(resized_img, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)

# Display the images
cv.imshow("Original with Lines", resized_img)
cv.imshow("Mask", mask)
cv.imshow("Inpainted Image", inpainted_img)

# Save the results
cv.imwrite("../results/original_with_lines.jpg", resized_img)
cv.imwrite("../results/mask.jpg", mask)
cv.imwrite("../results/inpainted_image.jpg", inpainted_img)

cv.waitKey(0)
cv.destroyAllWindows()