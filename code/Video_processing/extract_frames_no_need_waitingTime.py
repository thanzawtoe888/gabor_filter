import cv2
import os

def extract_frames_fast(video_path, output_folder, interval=30):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps

    saved_count = 0
    current_time = 0

    while current_time < duration_sec:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)  # set time in milliseconds
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_count += 1
        current_time += interval

    cap.release()
    print(f"Frames extracted every {interval} seconds: {saved_count}")

# Example usage
video_path = r'C:\Users\luca\Desktop\Git Repo\myMatLab\my_research\auto_detect_ref\output_720p.mp4'
# video_path = r'D:\Crack-Dataset\my_data\7_5_2025_final\VID_20250507_161408.mp4'
# output_folder = r'D:\Crack-Dataset\my_data\extract_frames\7_5_2025_final\day7_3'
output_folder = r'C:\Users\luca\Desktop\Git Repo\myMatLab\my_research\auto_detect_ref\frames_720_final'
extract_frames_fast(video_path, output_folder, interval=30)
