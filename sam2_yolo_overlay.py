# https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb

import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import pandas as pd
import csv
from sam2.build_sam import build_sam2_video_predictor

'''
This code is similar to sam2_fb.py
This scripts adds to it the ability to overlay the YOLO detections on top of the results of SAM2  
'''

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([1, 0, 0], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def calculate_center(mask):
    mask = np.squeeze(mask)
    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None
    center_x = (np.mean(x_indices))
    center_y = (np.mean(y_indices))
    return center_x, center_y

csv_data = []

csv_file_path = "sam_video_results.csv"

output_dir = Path("sam_yolo_overlay")
output_dir.mkdir(exist_ok=True)

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "./SAM_frames"

sam_object_centers = []

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))

inference_state = predictor.init_state(video_path=video_dir)

predictor.reset_state(inference_state)

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started
# box = np.array([213.28414916992188,549.037109375,292.9756774902344,616.1963500976562], dtype=np.float32) # test video 1 frame 3250 id 58
#box = np.array([395.71099853515625,512.266357421875,447.809326171875,591.19580078125], dtype=np.float32) # reverse test video 1 : frame 4201 id 73
# box = np.array([291.9360656738281, 558.0534057617188, 362.87957763671875, 629.9723510742188], dtype=np.float32) # test video 2
# box = np.array([134.40127563476562,85.608154296875,208.40676879882812,145.95773315429688], dtype=np.float32) # test video 2 : frame 4802 id 91

# box = np.array([387.17547607421875,515.2574462890625,446.7567138671875,599.8990478515625], dtype=np.float32) #rev id 1 _ orig id 3 -> res id 4
# box = np.array([291.0841369628906,417.09405517578125,363.7495422363281,473.75067138671875], dtype=np.float32) #  rev id 2 _ orig id 1 -> res id 2
box = np.array([491.833740234375,244.58056640625,535.9593505859375,330.1253662109375], dtype=np.float32) #  rev id 3 _ orig id 2 -> res id 1
#box = np.array([212.82928466796875,547.2567138671875,293.5916748046875,616.7015380859375], dtype=np.float32) #  id 3 -> res id 4
# box = np.array([148.3681640625,546.9521484375,202.6973876953125,614.1187744140625], dtype=np.float32) #  id 2 -> res id 1
# box = np.array([309.03173828125,560.7134399414062,380.7294921875,619.0231323242188], dtype=np.float32) #  id 1 -> res id 2

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.savefig(output_dir / f"{ann_frame_idx:04d}.png", bbox_inches='tight', dpi=300)
plt.close()  # Close the figure to free memory

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

for out_frame_idx, obj_masks in video_segments.items():
    for out_obj_id, mask in obj_masks.items():
        center = calculate_center(mask)
        if center is None :
            continue
        csv_data.append({
            "track_id": 1,
            "frame": out_frame_idx,
            "center_x": center[0],
            "center_y": center[1],
            "mask_size": mask.sum()
        })

with open(csv_file_path, mode="w", newline="") as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=["track_id", "frame", "center_x", "center_y", "mask_size"])
    csv_writer.writeheader()
    for row in csv_data:
        csv_writer.writerow(row)

print(f"Results saved to {csv_file_path}")   

frame_offset = 0


target_track_ids = [3, 8, 9, 14, 15, 16, 17, 19, 20]  # reverse
colors = ["magenta", "green", "yellow", "blue", "red", "orange", "purple", "brown"] 

#target_track_ids = [2, 14] # reverse
#colors = ["magenta", "green"]

#target_track_ids = [1, 4]
#colors = ["magenta", "green"] 

#target_track_ids = [2, 7, 9, 12, 15, 17, 20, 22, 23, 25]
#colors = ["magenta", "green", "yellow", "blue", "red", "orange", "purple", "brown"]  

#target_track_ids = [3, 6, 13]
#colors = ["magenta", "green", "yellow"] 

#target_track_ids = [1, 14, 2] # reverse
#colors = ["magenta", "green", "yellow"] 

#frame_offset = 4251 # test video 2
#target_track_ids = [73, 91] # test video 2
#colors = ["magenta", "green"] # test video 2

# frame_offset = 3250 # test video 1
#target_track_ids = [73, 58, 91] # test video 1
#colors = ["magenta", "green", "yellow"] # test video 1

data_path = "./sam_test_video1/reverse_tracked_detections_with_latent.csv"
detections = pd.read_csv(data_path)
detections['frame'] = detections['frame'].astype(int)
detections['track_id'] = detections['track_id'].astype(int)
# detections = detections[(detections['frame'] >= 3250) & (detections['frame'] <= 4250)] # test video 1
# detections = detections[(detections['frame'] >= 4000) & (detections['frame'] <= 5000)] # test video 2

yolo_object_centers = {track_id: [] for track_id in target_track_ids}


# render the segmentation results every few frames
vis_frame_stride = 1
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))

    # yolo_frame_detections = detections[detections['frame'] == (len(frame_names) + frame_offset - out_frame_idx)] # reverse
    yolo_frame_detections = detections[detections['frame'] == (frame_offset + out_frame_idx)]
    for i in range(len(target_track_ids)):
        track_id = target_track_ids[i]
        track_detections = yolo_frame_detections[yolo_frame_detections['track_id'] == track_id]
        
        if not track_detections.empty:
            x1, y1, x2, y2 = track_detections.iloc[0][['x1', 'y1', 'x2', 'y2']]
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            yolo_object_centers[track_id].append(center) 
        
        if yolo_object_centers[track_id]:
            x_coords, y_coords = zip(*yolo_object_centers[track_id])
            plt.plot(x_coords, y_coords, color=colors[i], linestyle="-", linewidth=0.5, marker="o", markersize=1)
            
            plt.scatter(x_coords[-1], y_coords[-1], color=colors[i], marker="o", s=50, edgecolor="white", linewidth=0.5)

    
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
        center = calculate_center(out_mask)
        
        if center is not None:
            sam_object_centers.append(center)
            
            if len(sam_object_centers) > 1:
                x_coords, y_coords = zip(*sam_object_centers)
                plt.plot(x_coords, y_coords, color="cyan", linestyle="-", linewidth=0.5, marker="o", markersize=1)

            plt.scatter(center[0], center[1], color="cyan", marker="o", s=50, edgecolor="white", linewidth=0.5)

    plt.savefig(output_dir / f"{out_frame_idx:04d}.png", bbox_inches='tight', dpi=300)
    plt.close()