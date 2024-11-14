import cv2
import os

def create_video_from_images(image_folder, output_video_path, fps=25):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    
    images.sort()
    
    if len(images) == 0:
        print("No images found in the folder.")
        return
    
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    
    if frame is None:
        print(f"Error reading image {first_image_path}")
        return
    
    height, width, layers = frame.shape
    size = (width, height)

    if output_video_path.endswith('.webm'):
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error reading image {image_path}, skipping...")
            continue

        out.write(frame)

    out.release()
    print(f"Video saved to {output_video_path}")

image_folder = './sam_yolo_overlay'
output_video_path = 'sam_video_1_result.mp4'
fps = 30

create_video_from_images(image_folder, output_video_path, fps)
