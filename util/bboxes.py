import os
import cv2

def draw_boxes(image_path, bbox_file, output_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    with open(bbox_file, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        try:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
            
            x_center = int(x_center * width)
            y_center = int(y_center * height)
            bbox_width = int(bbox_width * width)
            bbox_height = int(bbox_height * height)
            
            x_min = int(x_center - bbox_width / 2)
            y_min = int(y_center - bbox_height / 2)
            x_max = int(x_center + bbox_width / 2)
            y_max = int(y_center + bbox_height / 2)
            
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        except ValueError as e:
            print(f"Error processing line in {bbox_file}: {line}. Error: {e}")
    
    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

def process_folders(images_folder, labels_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(images_folder, filename)
            bbox_file = os.path.join(labels_folder, filename.replace(".jpg", ".txt"))
            
            if os.path.exists(bbox_file):
                output_path = os.path.join(output_folder, "output_" + filename)
                draw_boxes(image_path, bbox_file, output_path)
            else:
                print(f"Bounding box file not found for {filename}")

images_folder = "./dataset1/resized/train/images"  
labels_folder = "./dataset1/resized/train/labels"  
output_folder = "./dataset1/bounding_boxes/resized/train"  

process_folders(images_folder, labels_folder, output_folder)

