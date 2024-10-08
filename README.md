# ISP : Iterative Smartest Point

# Introduction
This idea came from some recordings I had, without any extrinsic calibration. While I think it's fun to align pointclouds by hand (it isn't), ICP wasn't working so well for some specific cases.
Please note that this slightly overengineered alignment method only works with 2 RGBD cameras and only one person in both frames.

The global idea is the following :
- Segment a person body and retrieve 2D keypoints in a couple of RGB frames
- Project these into the 3D pointcloud using the depthmaps
- Create clusters of neighbors for each joint in both keypoints
- Align one keypoint to the other using each subset of the pointcloud as if they were matching in each pointcloud

# Installation
`pip install -e .`

# Usage
Every data structure is in the `structures` module.  
In `isp/config.py`, add your checkpoints folder path. It should contain checkpoints for [YOLO](https://docs.ultralytics.com/models/yolov8/) pose and segmentation models.  
The rest is pretty straightforward, there's a pipeline already implemented in the `pipelines` submodule.