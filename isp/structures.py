import numpy as np

class RGBDFrame():
    def __init__(self, rgb:np.ndarray, depth:np.ndarray, K:np.ndarray) -> None:
        self._rgb = rgb
        self._depth = depth
        self._K = K
    
    @property
    def image(self):
        return self._rgb
    
    @property
    def depth(self):
        return self._depth
    
    @property
    def K(self):
        return self._K
    

class DualKinectFrameStack():
    def __init__(self, K_0:np.ndarray, K_1:np.ndarray) -> None:
        self._K0 = K_0
        self._K1 = K_1
        self._frames = []

    @property
    def K(self):
        return {'cam_0':self._K0, 'cam_1':self._K1}
    
    def add_frames(self, rgb_cam_0:np.ndarray, depth_cam_0:np.ndarray, rgb_cam_1:np.ndarray, depth_cam_1:np.ndarray):
        self._frames.append(
            {
                'frame_0':RGBDFrame(rgb_cam_0, depth_cam_0, self._K0),
                'frame_1':RGBDFrame(rgb_cam_1, depth_cam_1, self._K1)
            }
        )
    
    @property
    def frames(self):
        return self._frames
    
    def __repr__(self):
        return 'Framestack with %d frames'%len(self._frames)