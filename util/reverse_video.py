import cv2

def reverse_video(input_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare to write the reversed video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Read frames in reverse order and write to output video
    frames = []
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frames.append(frame)

    for frame in reversed(frames):
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Reversed video saved to {output_path}")

# Example usage
input_path = "sam_test_video.mp4"  # Replace with your input video path
output_path = "reversed_sam_test_video.mp4"  # Specify the output path
reverse_video(input_path, output_path)
