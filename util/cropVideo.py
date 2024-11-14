from moviepy.video.io.VideoFileClip import VideoFileClip

def trim_video(input_path, output_path, duration=60):
    with VideoFileClip(input_path) as video:
        trimmed_video = video.subclip(0, duration)
        trimmed_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

input_video = 'resized_test_video.mp4'
output_video = 'ten_seconds.mp4'
trim_video(input_video, output_video, 10)
