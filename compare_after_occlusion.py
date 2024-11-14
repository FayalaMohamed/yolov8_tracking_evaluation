import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine

'''
This file contains the code to compare the latent features of two tracks before and after an occlusion.
The function main takes the following arguments:
- csv_file: the path to the csv file containing the tracked detections with latent features
- frame_id: the frame id where the occlusion occurs
- start_track_id_1: the track id of the first track before the occlusion
- start_track_id_2: the track id of the second track before the occlusion
- end_track_id_1: the track id of the first track after the occlusion
- end_track_id_2: the track id of the second track after the occlusion
It prints the resemblance matrix of the latent features of the two tracks before and after the occlusion.
'''

def load_and_process_data(csv_file):
    df = pd.read_csv(csv_file)
    
    df['latent_feature'] = df['latent_feature'].apply(eval)
    
    latent_features = np.vstack(df['latent_feature'].values)
    
    return df, latent_features

def calculate_resemblance_matrix_old(df, latent_features, frame_id, track_id_1, track_id_2):
    start_frame = frame_id - 5
    end_frame = frame_id + 5

    df_filtered = df[(df['frame'].between(start_frame, end_frame)) & 
                     (df['track_id'].isin([track_id_1, track_id_2]))]

    grouped = df_filtered.groupby('track_id')

    features_track_1 = grouped.get_group(track_id_1)['latent_feature'].values
    features_track_2 = grouped.get_group(track_id_2)['latent_feature'].values

    resemblance_matrix = np.zeros((len(features_track_1), len(features_track_2)))

    for i, feature_1 in enumerate(features_track_1):
        for j, feature_2 in enumerate(features_track_2):
            resemblance_matrix[i, j] = 1 - cosine(feature_1, feature_2)

    return resemblance_matrix

def calculate_resemblance_matrix(df, latent_features, frame_id, start_track_id_1, start_track_id_2, end_track_id_1, end_track_id_2, before=10, after=25):
    start_frame = frame_id - before
    end_frame = frame_id + after

    df_start = df[(df['frame'] == start_frame) & 
                     (df['track_id'].isin([start_track_id_1, start_track_id_2]))]

    df_end = df[(df['frame'] == end_frame) & 
                     (df['track_id'].isin([end_track_id_1, end_track_id_2]))]

    grouped_start = df_start.groupby('track_id')
    features_start_1 = grouped_start.get_group(start_track_id_1)['latent_feature'].values
    features_start_2 = grouped_start.get_group(start_track_id_2)['latent_feature'].values

    for track_id, group in grouped_start:
        print(f"Track ID: {track_id}")
        print(group[['frame', 'latent_feature']])
        print()

    grouped_end = df_end.groupby('track_id')
    features_end_1 = grouped_end.get_group(end_track_id_1)['latent_feature'].values
    features_end_2 = grouped_end.get_group(end_track_id_2)['latent_feature'].values
    
    for track_id, group in grouped_end:
        print(f"Track ID: {track_id}")
        print(group[['frame', 'latent_feature']])
        print()

    features_start_1 = np.array(features_start_1)
    features_start_2 = np.array(features_start_2)
    features_end_1 = np.array(features_end_1)
    features_end_2 = np.array(features_end_2)
    resemblance_matrix = np.zeros((2, 2))

    resemblance_matrix[0, 0] = 1 - cosine(features_start_1[0], features_end_1[0])
    resemblance_matrix[0, 1] = 1 - cosine(features_start_1[0], features_end_2[0])
    resemblance_matrix[1, 0] = 1 - cosine(features_start_2[0], features_end_1[0])
    resemblance_matrix[1, 1] = 1 - cosine(features_start_2[0], features_end_2[0])

    return resemblance_matrix

def main(csv_file, frame_id, start_track_id_1, start_track_id_2, end_track_id_1, end_track_id_2):
    df, latent_features = load_and_process_data(csv_file)
    
    resemblance_matrix = calculate_resemblance_matrix(df, latent_features, frame_id, start_track_id_1, start_track_id_2, end_track_id_1, end_track_id_2)
    return resemblance_matrix

if __name__ == "__main__":
    csv_file = 'tracked_detections_with_latent.csv'
    frame_id = 4542
    start_track_id_1 = 3
    start_track_id_2 = 73
    end_track_id_1 = 89
    end_track_id_2 = 91
    
    resemblance_matrix = main(csv_file, frame_id, start_track_id_1, start_track_id_2, end_track_id_1, end_track_id_2)
    print(resemblance_matrix)

# 674 : 1+10 -> 1+10
# 781 : 1+10+17(10 detecte 2 fois) -> 1+10
# 787, 788, 790 une crevette detectee 2 fois 
# 794 : 1+10 ->1+18
# 805 : 1 change de position (rotation bizarre) et est detecte comme 10
# 1070 : 18 perdu pendant 2 frames et devient 20 apres
# 1800 et 2186 et 2226 et 2326 : 20+10 -> 20 perdu pendant qqs frame -> 20+10
# 2338 : 20+10 -> 20 perdu pendant qqs frame + est remplace par 35 pendant qqs frams -> arrive a revenir a 10+20
# 2475 : 10+3 -> apparition de 37 pendant qqs frames et 10 perdu -> 10+3
# 2760 and 2767 : 3+10 -> 10 is detected by 10 and 45 -> only 45 is kept
# 2981 : 3+45 -> 45 is lost for some frames -> 3+58
# 3761 : 58+63 -> both lost for some frames -> 72+73
# 4542 : 3+73 -> 3eme detection(89) rajoutee au debut + 73 et 3 perdus et ne reste que 89 pendant qqs frames -> 89+91
# 5032 : 91+99 -> 99 cahche derriere 91 + nouvelle detection 100 remplace les 2 + 91 retrouve -> 91+104     
