from __future__ import absolute_import, division

import random
import numpy as np
from albumentations.augmentations import functional as F
from albumentations.core.transforms_interface import DualTransform


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
        for_choice = np.array(has_masks[1])
        random_w_start = random.random()
        if len(for_choice) > 0:
            start_positions = np.random.choice(for_choice)
            random_w_start = (start_positions - self.width) / (masks.shape[1] - self.width) \
                         - (random.random() + 0.1) / 10.
        return {'h_start': 0,
                'w_start': max(0, random_w_start)
                }

    def get_transform_init_args_names(self):
        return ('height', 'width')

    @property
    def targets_as_params(self):
        return ['mask']
