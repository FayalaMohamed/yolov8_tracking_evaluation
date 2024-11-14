
import sys
import argparse
from ultralytics import YOLO
import cv2
import pandas as pd
import torch

'''
This script allows to run YOLO tracking on a video and save the tracking results to a CSV file.
It also allows to extract latent features from the YOLO backbone and save them along with the tracking results.
The script can be run with the following command:
python track.py <video_path> <weights_path> [--latent]
'''

def run_yolo_tracking(video_path, weights_path):
    model = YOLO(weights_path)

    results = model.track(source=video_path, save=True, persist=True)
    #, hide_labels=True, hide_conf=True, line_thickness=1
    
    print(f"Tracking completed.")

class YOLOWithLatentSpace(YOLO):
    def extract_features(self, x, layer_idx=11):
        # Ensure x is on the same device as the model
        x = x.to(self.device)
        # Run the preprocessing and backbone up to the specified layer
        x = self.model.model[0:layer_idx](x)
        return x

def track_with_latent_space(video_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = YOLOWithLatentSpace(weights_path)
    model.to(device)

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = 'tracked_output.avi'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    csv_file = 'tracked_detections_with_latent.csv'
    detection_data = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare frame for YOLO
        frame_resized = cv2.resize(frame, (640, 640))
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frame_tensor = frame_tensor.to(device)

        # Extract latent features
        with torch.no_grad():
            latent_features = model.extract_features(frame_tensor)

        results = model.track(frame, persist=True)

        if results and results[0].boxes:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = box.id.item() if box.id is not None else -1
                conf = box.conf.item()
                cls = box.cls.item()

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Extract corresponding latent feature
                feature_x = int(((x1 + x2) / 2 / 640) * latent_features.shape[3])
                feature_y = int(((y1 + y2) / 2 / 640) * latent_features.shape[2])
                feature_vector = latent_features[0, :, feature_y, feature_x].cpu().numpy()

                detection_data.append({
                    'frame': frame_idx,
                    'track_id': track_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': conf,
                    'class': cls,
                    'latent_feature': feature_vector.tolist()
                })
        
        out.write(frame)
        print(f"Processed frame {frame_idx}")
        frame_idx += 1

    cap.release()
    out.release()

    df = pd.DataFrame(detection_data)
    df.to_csv(csv_file, index=False)
    print(f"Tracking information with latent features saved to {csv_file}")


# not used (allows to get latent features at all levels of the backbone)
def track_with_latent_space_all_layers(video_path, weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = YOLOWithLatentSpace(weights_path)
    model.to(device)

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_path = 'tracked_output.avi'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    csv_file = 'tracked_detections_with_latent.csv'
    detection_data = []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Prepare frame for YOLO
        frame_resized = cv2.resize(frame, (640, 640))
        frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        frame_tensor = frame_tensor.to(device)

        # Extract latent features for the first 12 layers
        latent_features_list = []
        with torch.no_grad():
            for layer_idx in range(1, 12):
                latent_features = model.extract_features(frame_tensor, layer_idx=layer_idx)
                latent_features_list.append(latent_features)

        results = model.track(frame, persist=True)

        if results and results[0].boxes:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                track_id = box.id.item() if box.id is not None else -1
                conf = box.conf.item()
                cls = box.cls.item()

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Extract corresponding latent features for each layer and add them as separate columns
                feature_columns = {}
                for layer_idx, latent_features in enumerate(latent_features_list, start=1):
                    feature_x = int(((x1 + x2) / 2 / 640) * latent_features.shape[3])
                    feature_y = int(((y1 + y2) / 2 / 640) * latent_features.shape[2])
                    feature_vector = latent_features[0, :, feature_y, feature_x].cpu().numpy()
                    feature_columns[f'latent_feature_layer_{layer_idx}'] = feature_vector.tolist()

                detection_data.append({
                    'frame': frame_idx,
                    'track_id': track_id,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'confidence': conf,
                    'class': cls,
                    **feature_columns
                })

        out.write(frame)
        print(f"Processed frame {frame_idx}")
        frame_idx += 1

    cap.release()
    out.release()

    df = pd.DataFrame(detection_data)
    df.to_csv(csv_file, index=False)
    print(f"Tracking information with latent features saved to {csv_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 tracking on a video")
    parser.add_argument("video_path", type=str, help="Path to the input video")
    parser.add_argument("weights_path", type=str, help="Path to the YOLOv8 custom weights file")
    parser.add_argument("--latent", action="store_true", help="Use latent space feature extraction")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.latent:
        track_with_latent_space(args.video_path, args.weights_path)
    else:
        run_yolo_tracking(args.video_path, args.weights_path)

