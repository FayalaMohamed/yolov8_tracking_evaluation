import cv2
import numpy as np
import pandas as pd
import os

def visualize_ground_truth(video_path, gt_file_path, output_path):
    gt_data = pd.read_csv(gt_file_path, header=None, names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        
        frame_data = gt_data[gt_data['frame'] == frame_number]
        
        for _, row in frame_data.iterrows():
            x, y, w, h = int(row['bb_left']), int(row['bb_top']), int(row['bb_width']), int(row['bb_height'])
            track_id = int(row['id'])
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)

    cap.release()
    out.release()
    print(f"Visualization complete. Output saved to {output_path}")


video_path = './MOT20-01-collage.webm'
gt_file_path = './TrackEval/data/gt/mot_challenge/MOT20-train/MOT20-01/gt/gt.txt'
output_path = 'MOT20-01_ground_truth_visualization.mp4'

visualize_ground_truth(video_path, gt_file_path, output_path)