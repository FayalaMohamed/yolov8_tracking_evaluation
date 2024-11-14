import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
This script reads the resolved_identities.csv file and the sam_test_video.mp4 file and draws the tracks of the objects in the video.
(csv contains coordinates of the center of each object in each frame)
'''

df = pd.read_csv('resolved_identities.csv')

cap = cv2.VideoCapture('sam_test_video.mp4')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output_with_tracks.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

colors = {}
unique_ids = df['object_id'].unique()
for obj_id in unique_ids:
    colors[obj_id] = tuple(np.random.randint(0, 255, 3).tolist())

current_frame = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_data = df[df['frame'] == current_frame]
    
    for obj_id, obj_data in frame_data.groupby('object_id'):
        points = obj_data[['x_position', 'y_position']].values
        confidence = obj_data['confidence'].iloc[0]

        for i in range(len(points) - 1):
            start_point = tuple(points[i].astype(int))
            end_point = tuple(points[i + 1].astype(int))
            cv2.line(frame, start_point, end_point, colors[obj_id], 2)
            cv2.circle(frame, start_point, 3, colors[obj_id], -1) 
            
        last_point = tuple(points[-1].astype(int))
        cv2.circle(frame, last_point, 3, colors[obj_id], -1)

        cv2.putText(frame, f'{obj_id}:{confidence:.2f}', (last_point[0] + 5, last_point[1] - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[obj_id], 1, cv2.LINE_AA)

    out.write(frame)
    current_frame += 1

cap.release()
out.release()
cv2.destroyAllWindows()
