import os
import cv2

'''
This script resizes images and bounding boxes to a new size.
'''

def resize_image_and_adjust_bboxes(image_path, bbox_file, output_image_path, output_bbox_path, new_size=(640, 640)):
    image = cv2.imread(image_path)
    original_height, original_width, _ = image.shape
    
    resized_image = cv2.resize(image, new_size)
    new_width, new_height = new_size
    
    with open(bbox_file, 'r') as f:
        lines = f.readlines()

    with open(output_bbox_path, 'w') as out_file:
        for line in lines:
            line = line.strip()
            if not line:
                continue

            try:
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
                
                out_file.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")
            
            except ValueError as e:
                print(f"Error processing line in {bbox_file}: {line}. Error: {e}")

    cv2.imwrite(output_image_path, resized_image)
    print(f"Resized image saved to {output_image_path}")
    print(f"Updated bounding boxes saved to {output_bbox_path}")

def process_resize_and_bboxes(images_folder, labels_folder, output_images_folder, output_labels_folder, new_size=(640, 640)):
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(images_folder, filename)
            bbox_file = os.path.join(labels_folder, filename.replace(".jpg", ".txt"))

            if os.path.exists(bbox_file):
                output_image_path = os.path.join(output_images_folder, filename)
                output_bbox_path = os.path.join(output_labels_folder, filename.replace(".jpg", ".txt"))

                resize_image_and_adjust_bboxes(image_path, bbox_file, output_image_path, output_bbox_path, new_size)
            else:
                print(f"Bounding box file not found for {filename}")

images_folder = "./dataset2/valid/images"  
labels_folder = "./dataset2/valid/labels"  
output_images_folder = "./dataset2/resized/valid/images"  
output_labels_folder = "./dataset2/resized/valid/labels"  

process_resize_and_bboxes(images_folder, labels_folder, output_images_folder, output_labels_folder)

