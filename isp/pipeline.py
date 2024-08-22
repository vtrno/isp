from typing import Union

import numpy as np
import torch
import tqdm
from pytorch3d.loss import chamfer_distance
from yacs.config import CfgNode

from . import image, spatial, structures


def process_dual_cam(frame_0:structures.RGBDFrame, frame_1:structures.RGBDFrame, cfg:CfgNode) -> Union[np.ndarray, float]:
    """
    Aligns pointclouds from 2 views

    Parameters
    ----------
    frame_0 : structures.RGBDFrame
        Frame from camera 0
    frame_1 : structures.RGBDFrame
        Frame from camera 1
    """
    dp = image.DataProcessor(cfg)
    
    image_m = frame_0.image
    depth_m = frame_0.depth

    image_s = frame_1.image
    depth_s = frame_1.depth

    kp_m = dp.estimate_pose(image_m)
    kp_s = dp.estimate_pose(image_s)

    kp_m, kp_s = spatial._filter_kp(kp_m, kp_s)

    f_depth_m = spatial._scatter(depth_m, kp_m)
    f_depth_s = spatial._scatter(depth_s, kp_s)

    f_depth_m, f_depth_s = spatial._match_points(f_depth_m, f_depth_s, depth_m.shape)
    pc_m = np.asarray(dp.get_pc(f_depth_m, frame_0.K, stride = 1).points)
    pc_s = np.asarray(dp.get_pc(f_depth_s, frame_1.K, stride = 1).points)

    full_pc_m = np.asarray(dp.get_filtered_pc(image_m, depth_m, frame_0.K, stride = cfg.PROCESSING.STRIDE).points)
    full_pc_s = np.asarray(dp.get_filtered_pc(image_s, depth_s, frame_1.K, stride = cfg.PROCESSING.STRIDE).points)

    sub_m = spatial._create_opt_subset(full_pc_m, spatial._gather_subsets(pc_m, full_pc_m, cfg.PROCESSING.KNN_POINTS))
    sub_s = spatial._create_opt_subset(full_pc_s, spatial._gather_subsets(pc_s, full_pc_s, cfg.PROCESSING.KNN_POINTS))

    best_params, best_loss = None, float('inf')
    print('Round 1: Constrained optimization...')
    model = spatial.RigidTransform().to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    pbar = tqdm.tqdm(range(cfg.PROCESSING.MAX_STEPS))
    previous_loss = float('inf')
    for _ in pbar:
        opt.zero_grad()
        out = model(sub_s)
        loss = chamfer_distance(sub_m, out)[0]
        loss.backward()
        opt.step()
        pbar.set_description(str(round(loss.item(), 5)))
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = model.state_dict()
        
        if abs(loss.item() - previous_loss) < cfg.PROCESSING.EPSILON:
            print('Plateau reached, early stopping.')
            break
        previous_loss = loss.item()
        
    model.load_state_dict(best_params)
    best_params, best_loss = None, float('inf')
    previous_loss = float('inf')
    print('Round 2: Free optimization...')
    rot_pc = torch.tensor(full_pc_s.astype(np.float32)).unsqueeze(0).to(cfg.DEVICE)
    ref_pc = torch.tensor(full_pc_m.astype(np.float32)).unsqueeze(0).to(cfg.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    pbar = tqdm.tqdm(range(cfg.PROCESSING.MAX_STEPS))
    for _ in pbar:
        opt.zero_grad()
        out = model(rot_pc)
        loss = chamfer_distance(ref_pc, out)[0]
        loss.backward()
        opt.step()
        pbar.set_description(str(round(loss.item(), 5)))
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = model.state_dict()
        
        if abs(loss.item() - previous_loss) < cfg.PROCESSING.EPSILON:
            print('Plateau reached, early stopping.')
            break
        previous_loss = loss.item()
    model.load_state_dict(best_params)
    return model.transformation.cpu().numpy(), best_loss