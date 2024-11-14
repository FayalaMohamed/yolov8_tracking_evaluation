import cv2
import os

def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_filename = os.path.join(output_folder, f"{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    
    cap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'.")

video_path = './first_minute_resized_test_video.mp4'  
output_folder = './original_frames'
extract_frames(video_path, output_folder)
