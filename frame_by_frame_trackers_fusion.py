import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum, auto
import csv
import os

'''
This script is similar to fusion_trackers.py but it is used to find similar tracks by processing data frame by frame and finding the 
best object center for each object in the video
This work is not finished yet, it is still in progress
'''

class TrackingMethod(Enum):
    YOLO_FORWARD = auto()
    YOLO_REVERSE = auto()
    SAM_FORWARD = auto()
    SAM_REVERSE = auto()

    
@dataclass
class TrackingData:
    frame: int
    position: Tuple[float, float]
    confidence: float
    track_id: int  # Original track ID from the method

@dataclass
class FrameTrackData:
    frame: int
    method: TrackingMethod
    track_id: int
    position: Tuple[float, float]
    confidence: float

@dataclass
class TrackSegment:
    start_frame: int
    end_frame: int
    method: TrackingMethod
    track_id: int
    positions: List[Tuple[int, Tuple[float, float]]]  # List of (frame, position)
    confidences: List[float]

class MultiMethodTrackFusion:
    def __init__(self, 
                 num_objects: int = 3,
                 max_distance: float = 35.0,
                 min_overlap_frames: int = 1):
        self.num_objects = num_objects
        self.max_distance = max_distance
        self.min_overlap_frames = min_overlap_frames
        
        self.max_frame = 0

        self.method_tracks: Dict[TrackingMethod, Dict[int, List[TrackingData]]] = {
            method: {} for method in TrackingMethod
        }

        self.frames_data: Dict[int, List[FrameTrackData]] = {}

        self.tracking_results: Dict[int, Dict[int, Tuple[Tuple[float, float], float]]] = {}
        
        self.track_segments: List[TrackSegment] = []
        
        self.object_tracks: Dict[int, List[TrackSegment]] = {
            i: [] for i in range(num_objects)
        }

    def add_yolo_tracks(self, 
                       detections: pd.DataFrame,
                       method: TrackingMethod):
        if method not in [TrackingMethod.YOLO_FORWARD, TrackingMethod.YOLO_REVERSE]:
            raise ValueError("Invalid method for YOLO tracks")

        tracks_dict = self.method_tracks[method]
        
        for track_id in detections['track_id'].unique():
            track_dets = detections[detections['track_id'] == track_id]
            
            if len(track_dets) == 0:
                continue
            
            # Store track data
            positions = []
            confidences = []
            frames = []
            
            for _, det in track_dets.iterrows():
                frame = int(det['frame'])
                center = ((det['x1'] + det['x2']) / 2, (det['y1'] + det['y2']) / 2)
                confidence = det['confidence']
                
                if frame > self.max_frame:
                    self.max_frame = frame

                if track_id not in tracks_dict:
                    tracks_dict[track_id] = []
                
                tracks_dict[track_id].append(TrackingData(
                    frame=frame,
                    position=center,
                    confidence=confidence,
                    track_id=track_id
                ))

                self.frames_data[frame] = self.frames_data.get(frame, []) + [FrameTrackData(
                    frame=frame,
                    method=method,
                    track_id=track_id,
                    position=center,
                    confidence=confidence
                )]
                
                positions.append((frame, center))
                confidences.append(confidence)
                frames.append(frame)
            
            self.track_segments.append(TrackSegment(
                start_frame=min(frames),
                end_frame=max(frames),
                method=method,
                track_id=track_id,
                positions=sorted(positions),
                confidences=confidences
            ))

    def add_sam_tracks(self, 
                      frames_data: Dict[int, Dict[int, Tuple[Tuple[float, float], float]]],
                      method: TrackingMethod):
        if method not in [TrackingMethod.SAM_FORWARD, TrackingMethod.SAM_REVERSE]:
            raise ValueError("Invalid method for SAM tracks")

        tracks_dict = self.method_tracks[method]
        
        # First, identify all unique track IDs
        track_ids = set()
        for frame_data in frames_data.values():
            track_ids.update(frame_data.keys())
        
        for track_id in track_ids:
            positions = []
            confidences = []
            frames = []
            
            for frame, frame_data in frames_data.items():
                if frame > self.max_frame:
                    self.max_frame = frame

                if track_id in frame_data:
                    center, mask_size = frame_data[track_id]
                    confidence = 1.0  # TODO: Improve confidence calculation
                    
                    if track_id not in tracks_dict:
                        tracks_dict[track_id] = []
                    
                    tracks_dict[track_id].append(TrackingData(
                        frame=frame,
                        position=center,
                        confidence=confidence,
                        track_id=track_id
                    ))

                    self.frames_data[frame] = self.frames_data.get(frame, []) + [FrameTrackData(
                        frame=frame,
                        method=method,
                        track_id=track_id,
                        position=center,
                        confidence=confidence
                    )]
                    
                    positions.append((frame, center))
                    confidences.append(confidence)
                    frames.append(frame)
            
            if frames: 
                self.track_segments.append(TrackSegment(
                    start_frame=min(frames),
                    end_frame=max(frames),
                    method=method,
                    track_id=track_id,
                    positions=sorted(positions),
                    confidences=confidences
                ))

    def _calculate_segment_similarity(self, segment1: TrackSegment, segment2: TrackSegment, start_frame: int = None, end_frame: int = None) -> float:
        start_frame = max(segment1.start_frame, segment2.start_frame)
        end_frame = min(segment1.end_frame, segment2.end_frame)
        
        if end_frame - start_frame + 1 < self.min_overlap_frames:
            return 0.0
        
        pos1_dict = dict(segment1.positions)
        pos2_dict = dict(segment2.positions)
        
        common_frames = set(pos1_dict.keys()) & set(pos2_dict.keys())
        if not common_frames:
            return 0.0
            
        distances = []
        for frame in common_frames:
            if start_frame and end_frame and not frame in range(start_frame, end_frame + 1):
                continue 
            pos1 = np.array(pos1_dict[frame])
            pos2 = np.array(pos2_dict[frame])
            dist = np.linalg.norm(pos1 - pos2)
            distances.append(dist)

        if not distances:
            return 0.0

        avg_distance = np.mean(distances)
        similarity = np.exp(-avg_distance / self.max_distance)
        
        return similarity

    def resolve_identities(self):
        WINDOW_SIZE = 5
    
        resolved_tracks = defaultdict(dict)  # frame -> object_id -> (position, confidence)
        
        track_to_object_mapping = {}  # YOLO_FORWARD track_id -> object_id

        method_weight = {
            TrackingMethod.SAM_FORWARD: 0.9,
            TrackingMethod.YOLO_FORWARD: 0.8,
            TrackingMethod.SAM_REVERSE: 0.7,
            TrackingMethod.YOLO_REVERSE: 0.6,
        }
        
        for frame in range(self.max_frame + 1):
            start_frame = max(0, frame - WINDOW_SIZE)
            end_frame = min(self.max_frame, frame + WINDOW_SIZE)
            
            active_segments = []
            for segment in self.track_segments:
                if segment.start_frame <= frame <= segment.end_frame:
                    active_segments.append(segment)
            
            if frame == 0:
                yolo_forward_tracks = [s for s in active_segments if s.method == TrackingMethod.YOLO_FORWARD]
                
                for obj_id, track in enumerate(yolo_forward_tracks[:self.num_objects]):
                    track_to_object_mapping[track.track_id] = obj_id
            
            n_segments = len(active_segments)
            similarity_matrix = np.zeros((n_segments, n_segments))
            
            for i in range(n_segments):
                for j in range(i + 1, n_segments):
                    similarity = self._calculate_segment_similarity(
                        active_segments[i],
                        active_segments[j],
                        start_frame,
                        end_frame
                    )
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

            track_groups = []
            processed = set()
            
            for i in range(n_segments):
                if i in processed:
                    continue
                    
                group = {i}
                processed.add(i)
                
                for j in range(n_segments):
                    if j not in processed and similarity_matrix[i, j] > 0.7:
                        group.add(j)
                        processed.add(j)
                        
                track_groups.append(group)
            
            frame_positions = {}
            
            for obj_id in range(self.num_objects):
                best_position = None
                best_confidence = 0
                
                for group in track_groups:
                    group_segments = [active_segments[i] for i in group]

                    group_matches_object = False
                    for segment in group_segments:
                        if (segment.method == TrackingMethod.YOLO_FORWARD and 
                            segment.track_id in track_to_object_mapping and 
                            track_to_object_mapping[segment.track_id] == obj_id):
                            group_matches_object = True
                            break
                    
                    if not group_matches_object:
                        continue
                    
                    positions = []
                    weights = []
                    
                    for segment in group_segments:
                        pos_dict = dict(segment.positions)
                        if frame in pos_dict:
                            positions.append(np.array(pos_dict[frame]))
                            
                            conf_idx = segment.positions.index((frame, pos_dict[frame]))
                            weight = segment.confidences[conf_idx] * method_weight[segment.method]
                            weights.append(weight)
                            
                    if positions:
                        avg_position = np.average(positions, weights=weights, axis=0)
                        avg_confidence = np.mean(weights)
                        
                        if avg_confidence > best_confidence:
                            best_position = tuple(avg_position)
                            best_confidence = avg_confidence
                
                if best_position is not None:
                    frame_positions[obj_id] = (best_position, best_confidence)
            
            resolved_tracks[frame] = frame_positions
            
            if frame < self.max_frame:
                next_frame_segments = [s for s in self.track_segments 
                                    if s.method == TrackingMethod.YOLO_FORWARD and 
                                    s.start_frame <= frame + 1 <= s.end_frame]
                
                for segment in next_frame_segments:
                    if segment.track_id not in track_to_object_mapping:
                        pos_dict = dict(segment.positions)
                        if frame + 1 in pos_dict:
                            track_pos = np.array(pos_dict[frame + 1])
                            best_dist = float('inf')
                            best_obj_id = None
                            
                            for obj_id, (obj_pos, _) in frame_positions.items():
                                dist = np.linalg.norm(track_pos - np.array(obj_pos))
                                if dist < best_dist :
                                    best_dist = dist
                                    best_obj_id = obj_id
                            
                            if best_obj_id is not None:
                                track_to_object_mapping[segment.track_id] = best_obj_id
        
        self.tracking_results = resolved_tracks
        
        print("\nYOLO Forward track_id to object_id mappings:")
        for track_id, obj_id in sorted(track_to_object_mapping.items()):
            print(f"Track {track_id} -> Object {obj_id}")
        
        self.save_resolved_identities_to_csv(resolved_tracks)

        return resolved_tracks, track_to_object_mapping
    
    def save_resolved_identities_to_csv(self, resolved_tracks: Dict[int, Dict[int, Tuple]], output_path: str = "resolved_identities.csv"):
        csv_rows = []
        
        for frame, frame_data in sorted(resolved_tracks.items()):
            for object_id, (position, confidence) in frame_data.items():
                x_pos, y_pos = position
                
                row = {
                    'frame': frame,
                    'object_id': object_id,
                    'x_position': x_pos,
                    'y_position': y_pos,
                    'confidence': confidence
                }
                csv_rows.append(row)
        
        fieldnames = ['frame', 'object_id', 'x_position', 'y_position', 'confidence']
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"Saved resolved identities to: {os.path.abspath(output_path)}")
            


def prepare_yolo_forward_tracks(csv_path, reverse=False):
    detections = pd.read_csv(csv_path)
    detections['frame'] = detections['frame'].astype(int)
    detections['track_id'] = detections['track_id'].astype(int)
    if reverse :
        detections['frame'] = 951 - detections['frame']

    if 'latent_feature' in detections.columns:
        detections = detections.drop(columns=['latent_feature'])
    
    return detections.copy()


def prepare_sam_tracks(csv_path: str, reverse=False):
    sam_data = pd.read_csv(csv_path)
    sam_data['frame'] = sam_data['frame'].astype(int)
    sam_data['track_id'] = sam_data['track_id'].astype(int)
    sam_data['center_x'] = sam_data['center_x'].astype(float)
    sam_data['center_y'] = sam_data['center_y'].astype(float)
    sam_data['mask_size'] = sam_data['mask_size'].astype(int)
    
    if reverse:
        sam_data['frame'] = 951 - sam_data['frame']

    processed_segments = {}
    for _, row in sam_data.iterrows():
        frame_idx = row['frame']
        track_id = row['track_id']
        center = (row['center_x'], row['center_y'])
        mask_size = row['mask_size']

        if frame_idx not in processed_segments:
            processed_segments[frame_idx] = {}
        processed_segments[frame_idx][track_id] = (center, mask_size)
            
    return processed_segments


detections_forward = prepare_yolo_forward_tracks('sam_test_video1/tracked_detections_with_latent.csv')
detections_reverse = prepare_yolo_forward_tracks('sam_test_video1/reverse_tracked_detections_with_latent.csv', reverse=True)

video_segments_forward_1 = prepare_sam_tracks('sam_test_video1/sam_video_results_id1.csv')
video_segments_reverse_1 = prepare_sam_tracks('sam_test_video1/reverse_sam_video_results_id1_orig_id2_rev.csv', reverse=True)
video_segments_forward_2 = prepare_sam_tracks('sam_test_video1/sam_video_results_id2.csv')
video_segments_reverse_2 = prepare_sam_tracks('sam_test_video1/reverse_sam_video_results_id2_orig_id3_rev.csv', reverse=True)
video_segments_forward_3 = prepare_sam_tracks('sam_test_video1/sam_video_results_id3.csv')
video_segments_reverse_3 = prepare_sam_tracks('sam_test_video1/reverse_sam_video_results_id3_orig_id1_rev.csv', reverse=True)


# Initialize the system
fusion = MultiMethodTrackFusion(num_objects=3)

# Add tracks from each method separately
fusion.add_yolo_tracks(detections_forward, TrackingMethod.YOLO_FORWARD)
fusion.add_yolo_tracks(detections_reverse, TrackingMethod.YOLO_REVERSE)
fusion.add_sam_tracks(video_segments_forward_1, TrackingMethod.SAM_FORWARD)
fusion.add_sam_tracks(video_segments_reverse_1, TrackingMethod.SAM_REVERSE)
fusion.add_sam_tracks(video_segments_forward_2, TrackingMethod.SAM_FORWARD)
fusion.add_sam_tracks(video_segments_reverse_2, TrackingMethod.SAM_REVERSE)
fusion.add_sam_tracks(video_segments_forward_3, TrackingMethod.SAM_FORWARD)
fusion.add_sam_tracks(video_segments_reverse_3, TrackingMethod.SAM_REVERSE)

# Resolve true identities
fusion.resolve_identities()

# sam 1 : 
## reverse : 3, 8, 9, 14, 15, 16, 17, 19, 20
## forward : 2, 7, 9, 12, 15, 17, 20, 22, 23, 25

# sam 2 :
## reverse : 2, 14
## forward : 1, 4

# sam 4 :
## reverse : 1, 14, 2
## forward : 3, 6, 13