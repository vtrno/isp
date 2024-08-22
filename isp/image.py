import os

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO
from yacs.config import CfgNode


def read_data(datafile:str) -> dict:
    out = {}
    data = np.load(datafile)
    for k in data.keys():
        if isinstance(data[k].dtype, np.float64):
            out[k] = data[k].astype(np.float32)
        else:
            out[k] = data[k]
    return out

class DataProcessor():
    def __init__(self, config:CfgNode) -> None:
        # ? Find checkpoints
        files = [os.path.join(os.path.realpath(config.INFERENCE.CKPT_PATH), x) for x in os.listdir(os.path.realpath(config.INFERENCE.CKPT_PATH))]
        seg_ckpt = [x for x in files if 'seg' in x]
        pose_ckpt = [x for x in files if 'pose' in x]
        if len(seg_ckpt) != 1 or len(pose_ckpt) != 1:
            raise ValueError('Error while loading checkpoints, please make sure everything is set up correctly.')

        self._seg = YOLO(seg_ckpt[0])
        self._pose = YOLO(pose_ckpt[0])
        self._config = config

    def segment(self, image:np.ndarray, p_idx:int=0) -> np.ndarray:
        results = self._seg(image, verbose=False)
        return np.where(cv2.resize(results[0].masks.data.cpu()[results[0].boxes.cls.cpu() == 0][p_idx].numpy(), (image.shape[1], image.shape[0])) > 0.5, 1, 0).astype(bool)
    
    def estimate_pose(self, image:np.ndarray, p_idx:int=0) -> np.ndarray:
        results = self._pose(image, verbose=False)
        return results[0].keypoints.xy.cpu()[results[0].boxes.cls.cpu() == 0][p_idx].numpy()[self._config.PROCESSING.COCO_KP]

    def get_pc(self, depth, K, stride:int = None):
        fx = K[0, 0]
        fy = K[1, 1]
        cx, cy = K[:2, 2]
        image = o3d.geometry.Image(depth)
        camera = o3d.camera.PinholeCameraIntrinsic(max(depth.shape[:2]), min(depth.shape[:2]), fx, fy, cx, cy)
        return o3d.geometry.PointCloud.create_from_depth_image(image, camera, depth_trunc=self._config.PROCESSING.DEPTH_CUT, stride = self._config.PROCESSING.STRIDE if stride is None else stride)
    
    def get_filtered_pc(self, image, depth, K, stride:int=None):
        stride = self._config.PROCESSING.STRIDE if stride is None else stride
        f_d = self.segment(image) * depth
        return self.get_pc(f_d, K, stride)