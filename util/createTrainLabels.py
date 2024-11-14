import os

'''
This file is used to convert the MOT20 dataset to the format required by YOLO
(We used this to finetune YOLO on the MOT20 dataset)
'''

img_width = 1654
img_height = 1080  

image_dir = "./MOT20/train/MOT20-05/img1"
gt_dir = "./MOT20/train/MOT20-05/gt"
gt_file = os.path.join(gt_dir, "gt.txt")

output_dir = "./MOT20/train/MOT20-05/labels"

frame_data = {}

with open(gt_file, "r") as f:
    for line in f:
        frame_id, track_id, bb_left, bb_top, bb_width, bb_height, conf, _, _ = line.strip().split(",")

        x_center = (float(bb_left) + float(bb_width) / 2) / img_width
        y_center = (float(bb_top) + float(bb_height) / 2) / img_height
        width = float(bb_width) / img_width
        height = float(bb_height) / img_height

        frame_num = str(int(frame_id)).zfill(6)
        if frame_num not in frame_data:
            frame_data[frame_num] = []

        frame_data[frame_num].append(f"{track_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

for frame_num, bbox_list in frame_data.items():
    txt_filename = os.path.join(output_dir, f"{frame_num}.txt")
    with open(txt_filename, "w") as txt_file:
        txt_file.write("\n".join(bbox_list))
