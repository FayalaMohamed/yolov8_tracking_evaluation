import supervision as sv
''''
frames_generator = sv.get_video_frames_generator('sam_test_video.mp4')
sink = sv.ImageSink(
    target_dir_path='SAM_frames',
    image_name_pattern="{:05d}.jpeg")

with sink:
    for frame in frames_generator:
        sink.save_image(frame) '''

frames_generator = sv.get_video_frames_generator('./sam_test_video1/tracked_output.avi')
frames = [frame for frame in frames_generator]

sink = sv.ImageSink(
    target_dir_path='./sam_test_video1/test_orig',
    image_name_pattern="{:04d}.jpeg"
)

with sink:
    for i, frame in enumerate(frames):
        sink.save_image(frame)
