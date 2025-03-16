# import cv2
# import os

# # Function to extract frames from a video
# def extract_frames(video_path, output_folder):
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Load the video
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0

#     # Loop through each frame
#     while True:
#         ret, frame = cap.read()
        
#         if not ret:
#             break

#         # Save frame as an image
#         frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
#         cv2.imwrite(frame_filename, frame)

#         frame_count += 1

#     # Release the video capture object
#     cap.release()
#     print(f"Frames extracted: {frame_count}")

# # Example usage
# video_path = r'D:\Crack-Dataset\my_data\26_2_25_Bending_test_for_Calcined_clay\VID_20250226_133535.mp4'
# output_folder =r'C:\Users\luca\Desktop\New folder\extract_frames'
# extract_frames(video_path, output_folder)

import cv2
import os

# Function to extract frames from a video every given interval
def extract_frames(video_path, output_folder, interval=30):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = frame_rate * interval
    frame_count = 0
    saved_count = 0

    # Loop through each frame
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        # Save frame every 'interval' seconds
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Frames extracted every {interval} seconds: {saved_count}")

# Example usage
    
video_path = r'D:\Crack-Dataset\my_data\26_2_25_Bending_test_for_Calcined_clay\day1_3.mp4'
output_folder = r'D:\Crack-Dataset\my_data\extract_frames\26_2_25_Bending_test_for_Calcined_clay\day1_3'
extract_frames(video_path, output_folder, interval=30)
