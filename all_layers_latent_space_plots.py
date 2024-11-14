import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import plotly.express as px
import plotly.graph_objects as go

'''
This file is similar to latent_space.py but it creates the plots for the latent space features of different layers.
'''

def load_and_process_data(csv_file, idx=11):
    df = pd.read_csv(csv_file)
    
    df[f'latent_feature_layer_{idx}'] = df[f'latent_feature_layer_{idx}'].apply(eval)
    
    latent_features = np.vstack(df[f'latent_feature_layer_{idx}'].values)
    
    return df, latent_features

def apply_pca(latent_features, n_components=3):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(latent_features)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    return pca_result

def plot_3d_image(pca_result, df, idx=11):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    unique_tracks = df['track_id'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_tracks)))
    
    for track, color in zip(unique_tracks, colors):
        mask = df['track_id'] == track
        ax.scatter(pca_result[mask, 0], pca_result[mask, 1], pca_result[mask, 2], 
                   c=[color], label=f'Track {track}')

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('PCA of Latent Space Features')
    
    ax.legend()
    plt.savefig(f'./figs/latent_space_pca_layer_{idx}.png')
    plt.show()


def plot_3d_interactive_plotly(pca_result, df, idx=11):
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
    pca_df['track_id'] = df['track_id'].values

    unique_tracks = pca_df['track_id'].unique()
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_tracks))) 
    color_map = {track_id: f'rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})'
                 for track_id, color in zip(unique_tracks, colors)}
    
    fig = go.Figure()

    for track in unique_tracks:
        track_data = pca_df[pca_df['track_id'] == track]
        
        fig.add_trace(go.Scatter3d(
            x=track_data['PC1'],
            y=track_data['PC2'],
            z=track_data['PC3'],
            mode='lines+markers',
            line=dict(color=color_map[track], width=2),
            marker=dict(
                size=5,
                line=dict(color=color_map[track], width=2),
            ),
            name=f'Track {track}'
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        ),
        title='PCA of Latent Space Features',
        legend_title='Track ID'
    )

    fig.write_html(f'./figs/interactive_plot_layer_{idx}.html')

    fig.show()


def main(csv_file):
    nb_frames = 100
    for idx in range(1,12):
        df, latent_features = load_and_process_data(csv_file, idx)
        
        df_filtered = df[df['frame'].isin(range(nb_frames))]
        latent_features_filtered = latent_features[df['frame'].isin(range(nb_frames))]

        track_counts = df_filtered['track_id'].value_counts()
        valid_tracks = track_counts[track_counts > 20].index 

        df_filtered = df_filtered[df_filtered['track_id'].isin(valid_tracks)]
        latent_features_filtered = latent_features[df_filtered.index]
        
        pca_result = apply_pca(latent_features_filtered)
        
        plot_3d_image(pca_result, df_filtered, idx)

        plot_3d_interactive_plotly(pca_result, df_filtered, idx)

if __name__ == "__main__":
    csv_file = 'tracked_detections_with_all_latent.csv'
    main(csv_file)
