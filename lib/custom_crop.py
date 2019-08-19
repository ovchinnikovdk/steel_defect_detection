from __future__ import absolute_import, division

from types import LambdaType
import math
import random
import warnings

import cv2
import numpy as np

from albumentations.augmentations import functional as F
from albumentations.augmentations.bbox_utils import union_of_bboxes, denormalize_bbox, normalize_bbox
from albumentations.core.transforms_interface import to_tuple, DualTransform, ImageOnlyTransform, NoOp
from albumentations.core.utils import format_args


class CustomCrop(DualTransform):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super(CustomCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, h_start=0, w_start=0, **params):
        return F.random_crop(img, self.height, self.width, h_start, w_start)

    def get_params_dependent_on_targets(self, params):
        masks = params['mask']
        has_masks = np.where(masks > 0)
        start_positions = np.random.choice(list(zip(has_masks[0], has_masks[1])))
        return {'h_start': 0, 'w_start': min(0, start_positions[1] / masks.shape[1] - (random.random()) / 10.)}

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_random_crop(bbox, self.height, self.width, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_random_crop(keypoint, self.height, self.width, **params)

    def get_transform_init_args_names(self):
        return ('height', 'width')
