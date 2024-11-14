import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum, auto

'''
This script reads the tracked_detections_with_latent.csv file and the sam_video_results_id1.csv, sam_video_results_id2.csv, 
sam_video_results_id3.csv files and resolves the identities of the objects in the video.
The idea is that we are using YOLOv8 and SAM2 to track all the shrimps in the video and the reverse chronoligical order of the video.
We are then using the MultiMethodTrackFusion class to resolve the identities of the shrimps. THis is done by putting similar tracks in the same bucket.
And because we know the number of unique shrimps in the video, we can force the number of final buckets and assign the tracks to the shrimps.
The similarity between two tracks is calculated by the mean of the  euclidean distance between the centers of the tracks over all the frames in which they both exist. 
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
class TrackSegment:
    start_frame: int
    end_frame: int
    method: TrackingMethod
    track_id: int  # Original track ID from the method
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
        
        self.method_tracks: Dict[TrackingMethod, Dict[int, List[TrackingData]]] = {
            method: {} for method in TrackingMethod
        }
        
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
                
                if track_id not in tracks_dict:
                    tracks_dict[track_id] = []
                
                tracks_dict[track_id].append(TrackingData(
                    frame=frame,
                    position=center,
                    confidence=confidence,
                    track_id=track_id
                ))
                
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

    def _find_overlapping_segments(self, segment: TrackSegment) -> List[TrackSegment]:
        overlapping = []
        for other in self.track_segments:
            if other == segment:
                continue
                
            if (other.start_frame <= segment.end_frame and 
                other.end_frame >= segment.start_frame):
                overlapping.append(other)
        return overlapping

    def _calculate_segment_similarity(self, segment1: TrackSegment, segment2: TrackSegment) -> float:
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
            pos1 = np.array(pos1_dict[frame])
            pos2 = np.array(pos2_dict[frame])
            dist = np.linalg.norm(pos1 - pos2)
            distances.append(dist)
            
        avg_distance = np.mean(distances)
        similarity = np.exp(-avg_distance / self.max_distance)
        
        return similarity

    def _calculate_group_similarity(self, segment: TrackSegment, group: List[TrackSegment]) -> float:
        similarities = [self._calculate_segment_similarity(segment, g) for g in group]
        return max(similarities) if similarities else 0.0

    def resolve_identities(self):
        for method, tracks in self.method_tracks.items():
            print(f"{method.name}: {len(tracks)} tracks")

        yolo_segments = [s for s in self.track_segments if s.method in [TrackingMethod.YOLO_FORWARD, TrackingMethod.YOLO_REVERSE]]
        sam_segments = [s for s in self.track_segments if s.method in [TrackingMethod.SAM_FORWARD, TrackingMethod.SAM_REVERSE]]
        sorted_yolo_segments = sorted(
            yolo_segments,
            key=lambda s: (s.start_frame, -(s.end_frame - s.start_frame))
        )
        sorted_sam_segments = sorted(
            sam_segments,
            key=lambda s: (s.start_frame, -(s.end_frame - s.start_frame))
        )
        
        sorted_segments = sorted_sam_segments + sorted_yolo_segments

        identity_groups: List[List[TrackSegment]] = []
        
        for segment in sorted_segments:
            print(f"\nProcessing segment {segment.method} {segment.track_id} - {segment.start_frame} to {segment.end_frame}")
            if any(segment in group for group in identity_groups):
                continue
                
            overlapping = self._find_overlapping_segments(segment)
            
            similarities = [(other, self._calculate_segment_similarity(segment, other))
                          for other in overlapping]
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            added_to_existing = False
            for group in identity_groups:
                group_similarities = [self._calculate_segment_similarity(segment, g)
                                   for g in group]
                avg_similarity = np.mean(group_similarities) if group_similarities else 0
                print(group_similarities)
                print(avg_similarity)
                
                #if avg_similarity > 0.15:  # Threshold for group membership
                if any(s > 0.15 for s in group_similarities):
                    group.append(segment)
                    added_to_existing = True
                    break
            
            if not added_to_existing:
                identity_groups.append([segment])
        
        identity_groups.sort(key=lambda g: sum(s.end_frame - s.start_frame for s in g),
                           reverse=True)
        
        for i, group in enumerate(identity_groups):
            print("\n")
            print(f"Object {i + 1}: {len(group)} segments")
            for segment in sorted(group, key=lambda s: (s.method.name, s.track_id)):
                print(f"  {segment.method} {segment.track_id} - {segment.start_frame} to {segment.end_frame}")
        
        ''' for i, group in enumerate(identity_groups[:self.num_objects]):
            self.object_tracks[i] = group '''

        main_groups = identity_groups
        if len(main_groups) > self.num_objects:
            remaining_groups = main_groups[self.num_objects:]
            main_groups = main_groups[:self.num_objects]
            
            print("\nReassigning remaining tracks:")
            for group in remaining_groups:
                for segment in group:
                    group_similarities = [(i, self._calculate_group_similarity(segment, main_group))
                                       for i, main_group in enumerate(main_groups)]
                    
                    group_similarities.sort(key=lambda x: x[1], reverse=True)
                    
                    best_group_idx, best_similarity = group_similarities[0]
                    if best_similarity > 0.15:
                        main_groups[best_group_idx].append(segment)
                        print(f"  Assigned {segment.method} {segment.track_id} to Object {best_group_idx} "
                              f"(similarity: {best_similarity:.3f})")
        
        for i, group in enumerate(main_groups):
            self.object_tracks[i] = group

        for i, group in enumerate(main_groups):
            print("\n")
            print(f"Object {i + 1}: {len(group)} segments")
            for segment in sorted(group, key=lambda s: (s.method.name, s.track_id)):
                print(f"  {segment.method} {segment.track_id} - {segment.start_frame} to {segment.end_frame}")


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