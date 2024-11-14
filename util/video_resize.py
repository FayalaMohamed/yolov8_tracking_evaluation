from moviepy.editor import VideoFileClip

input_video_path = '/mnt/gammarus/20210608-Mathilde/MVI_0238-01.avi' 
output_video_path = 'resized_test_video.mp4'  

clip = VideoFileClip(input_video_path)

resized_clip = clip.resize(newsize=(640, 640))

resized_clip.write_videofile(output_video_path, codec='libx264')

print(f"Video resized and saved to {output_video_path}")

