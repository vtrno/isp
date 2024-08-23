from yacs.config import CfgNode as CN


_C = CN()
_C.DEVICE = 'cuda'

_C.INFERENCE = CN()
_C.INFERENCE.CKPT_PATH = None


_C.PROCESSING = CN()
_C.PROCESSING.DEPTH_CUT = 6.
_C.PROCESSING.COCO_KP = [0, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16]
_C.PROCESSING.KNN_POINTS = 100
_C.PROCESSING.STRIDE = 2
_C.PROCESSING.MAX_STEPS = 1000
_C.PROCESSING.EPSILON = 0.


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project. From YACS examples"""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
