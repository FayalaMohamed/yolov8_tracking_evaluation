import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

'''
This script reads the tracked_detections_with_latent.csv file and the original_frames folder and draws the trajectories of the objects in the video.
(csv contains coordinates of the bounding box of each object in each frame)
'''

data_path = "tracked_detections_with_latent.csv"
detections = pd.read_csv(data_path)
detections['frame'] = detections['frame'].astype(int)
detections['track_id'] = detections['track_id'].astype(int)
detections = detections[(detections['frame'] >= 4000) & (detections['frame'] <= 5000)]

target_track_ids = [73, 91]
colors = ["magenta", "green"]

video_dir = "original_frames" 
output_dir = "yolo_trajectories"
os.makedirs(output_dir, exist_ok=True)

object_centers = {track_id: [] for track_id in target_track_ids}

def calculate_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

for frame_id in sorted(detections['frame'].unique(), key=lambda x: int(x)):
    frame_detections = detections[detections['frame'] == frame_id]
    
    frame_name = f"{int(frame_id):04d}.jpg"
    frame_path = os.path.join(video_dir, frame_name)
    frame_image = Image.open(frame_path)
    
    plt.figure(figsize=(6, 4))
    plt.title(f"Frame {frame_id}")
    plt.imshow(frame_image)
    
    for i in range(len(target_track_ids)):
        track_id = target_track_ids[i]
        track_detections = frame_detections[frame_detections['track_id'] == track_id]
        
        if not track_detections.empty:
            x1, y1, x2, y2 = track_detections.iloc[0][['x1', 'y1', 'x2', 'y2']]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            object_centers[track_id].append(center) 
        
        if object_centers[track_id]:
            x_coords, y_coords = zip(*object_centers[track_id])
            plt.plot(x_coords, y_coords, color=colors[i], linestyle="-", linewidth=0.5, marker="o", markersize=1)
            
            plt.scatter(x_coords[-1], y_coords[-1], color=colors[i], marker="o", s=50, edgecolor="white", linewidth=0.5)
    
    output_frame_path = os.path.join(output_dir, f"{int(frame_id):04d}.png")
    plt.savefig(output_frame_path, bbox_inches='tight', dpi=300)
    plt.close()
