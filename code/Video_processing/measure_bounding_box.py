import cv2
import matplotlib.pyplot as plt

# Load the image
image_path = '../images/frame_0013.jpg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# Define the initial bounding box coordinates
box_top = int(height * 0.2)
box_bottom = int(height * 0.8)
box_left = int(width * 0.1)
box_right = int(width * 0.9)

# Draw the bounding box
cv2.rectangle(image, (box_left, box_top), (box_right, box_bottom), (0, 255, 0), 2)

# Draw the middle horizontal line
middle_y = height // 2
cv2.line(image, (0, middle_y), (width, middle_y), (0, 0, 255), 2)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Optionally, save the result
cv2.imwrite('output_with_box.png', image)
