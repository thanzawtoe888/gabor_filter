# import cv2
# import matplotlib.pyplot as plt
# import numpy as np

# # Load the image
# image_path = '../images/frame_0013.jpg'
# image = cv2.imread(image_path)
# height, width, _ = image.shape

# # Define the four points of the bounding box
# points = [(0.6, 775),(3833, 1496), (0.6, 1496), (3833, 775)  ]

# # Convert points to a numpy array
# pts = np.array(points, np.int32)
# pts = pts.reshape((-1, 1, 2))

# # Draw the bounding box
# cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

# # # Draw the middle horizontal line
# # middle_y = height // 2
# # cv2.line(image, (0, middle_y), (width, middle_y), (0, 0, 255), 2)

# # Display the image
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()

# # Save the output image
# cv2.imwrite('output_with_custom_box.png', image)

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = '../images/frame_0013.jpg'
image = cv2.imread(image_path)
height, width, _ = image.shape

# Define the four points of the bounding box
points = [(0, 775), (3833, 775), (3833, 1496), (0, 1496)]

# Convert points to a numpy array
pts = np.array(points, np.int32)
pts = pts.reshape((-1, 1, 2))

# Draw the bounding box
cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

# Display the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Save the output image
cv2.imwrite('output_with_custom_box.png', image)