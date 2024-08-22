import numpy as np
import roma
import torch
from pytorch3d.ops import knn_points


def _scatter(image:np.ndarray, keypoints:np.ndarray) -> np.ndarray:
    """
    Extracts keypoints for a certain depth image, keeping only the selected ones

    Parameters
    ----------
    image : np.ndarray
        Depth image, np.uint16
    keypoints : np.ndarray
        Keypoints coordinates

    Returns
    -------
    np.ndarray
        The scattered image
    """
    # ! This is highly suboptimal and needs more care but this is a POC so let's leave it like this for now
    kp = keypoints.astype(np.int64)
    filtered_kp = []
    pixel_values = []
    for x, y in kp:
        main_pixel_value = image[y, x]
        if (x>0 and y>0): # Check that keypoint is valid
            if main_pixel_value == 0:
                filtered_kp.append([0, 0])
                pixel_values.append(0)
            else:
                filtered_kp.append([x, y])
                pixel_values.append(main_pixel_value)

    filtered_kp = np.array(filtered_kp, dtype=np.int64)
    out = []
    for (x, y), p in zip(filtered_kp, pixel_values):
        out.append({'kp':(x,y),'depth':p})
    return out


def _filter_kp(kp_0, kp_1):
    kp_0_v = np.where(np.where(kp_0 > 0, 1, 0).sum(1) == 2, 1, 0)
    kp_1_v = np.where(np.where(kp_1 > 0, 1, 0).sum(1) == 2, 1, 0)
    valid_idx = (kp_0_v * kp_1_v).astype(bool)
    return kp_0[valid_idx], kp_1[valid_idx]

def _match_points(f_depth_m, f_depth_s, depth_shape):
    out0 = np.zeros(depth_shape, dtype=np.uint16)
    out1 = np.zeros(depth_shape, dtype=np.uint16)
    for p0, p1 in zip(f_depth_m, f_depth_s):
        if not (p0['depth'] == 0 or p1['depth'] == 0):
            out0[p0['kp'][1], p0['kp'][0]] = p0['depth']
            out1[p1['kp'][1], p1['kp'][0]] = p1['depth']
    return out0, out1

def _gather_subsets(keypoints:np.ndarray, pointcloud:np.ndarray, n_points:int=100, device:str='cuda'):
    pointcloud = torch.from_numpy(pointcloud.copy().astype(np.float32)).unsqueeze(0).to(device)
    keypoints = torch.from_numpy(keypoints.copy().astype(np.float32)).unsqueeze(0).to(device)
    return knn_points(keypoints, pointcloud, K=n_points).idx[0]

def _create_opt_subset(full_pc:np.ndarray, idxs:torch.Tensor):
    out = []
    torch_pc = torch.from_numpy(full_pc).float().to(idxs.device)
    for idx in idxs:
        out.append(torch_pc[idx])
    return torch.stack(out, 0)

class RigidTransform(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter('translation', torch.nn.Parameter(torch.zeros(3)))
        self.register_parameter('rotation', torch.nn.Parameter(torch.tensor([0., 0., 0., 1.])))
    
    @property
    def transformation(self):
        with torch.no_grad():
            rotmat = roma.unitquat_to_rotmat(self.rotation).cpu()
            transform = torch.cat([torch.cat([rotmat, self.translation.unsqueeze(1).cpu()], -1), torch.tensor([[0., 0., 0., 1.]])], 0)
        return transform

    def forward(self, x):
        """
        Applies rigid transformation to batched pointclouds

        Parameters
        ----------
        x : torch.Tensor
            Pointclouds of shape B, N, 3

        Returns
        -------
        torch.Tensor
            Pointclouds of shape B, N, 3
        """
        batch_size = x.shape[0]
        return torch.bmm(torch.stack([roma.unitquat_to_rotmat(self.rotation)]*batch_size), x.transpose(1, 2)).transpose(1, 2) + self.translation