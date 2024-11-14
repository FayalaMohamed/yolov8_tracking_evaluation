
This repository contains the code and some results of my work on the evaluation of YOLOv8 tracking capabilities on a custom dataset of shrimps in  apetri dish

# Requirements
- All libraries needed by this repository are in the `requirements.txt` file
- If you want to run any script using SAM2 you have to clone the SAM2 repository :
```
git clone https://github.com/facebookresearch/sam2.git && cd sam2

pip install -e .
``` 
- If you need to run the TrackEval repository to have MOT metrics you have to clone this repo :
```
git clone https://github.com/JonathonLuiten/TrackEval.git
```
You can run it on the MOT20-train dataset using this command : 
```
python scripts/run_mot_challenge.py --BENCHMARK MOT20 --TRACKERS_TO_EVAL YOLOv8x --TRACKERS_FOLDER ./data/trackers/mot_challenge --GT_FOLDER ./data/gt/mot_challenge
```
You also need to put the tracking result of YOLOv8x on the MOT20-train dataset (given by `mot_challenge.py`) in TrackEval/data/trackers/mot_challenge/MOT20-train/YOLOv8x/data
- Training YOLO on a custom dataset can be run with this command :
```
yolo task=detect mode=train model=yolov8x.pt data=./dataset/data.yaml epochs=100 plots=True imgsz=640
``` 
A data.yaml example for a dataset is : 
```
nc: 1
names: [shrimp]
test: test/images
train: train/images
val: valid/images
```
