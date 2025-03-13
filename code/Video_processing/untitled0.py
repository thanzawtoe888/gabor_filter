import cv2

# Path to video file
input_file = r'D:\Crack-Dataset\my_data\26_2_25_Bending_test_for_Calcined_clay\VID_20250226_133535.mp4'

# Open the video file
cap = cv2.VideoCapture(input_file)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# Play video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video finished or error in reading frame.")
        break
    
    cv2.imshow("Video Playback", frame)  # Show frame
    
    # Press 'q' to exit video playback
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
