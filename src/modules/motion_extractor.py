# coding: utf-8

"""
Motion extractor(M), which directly predicts the canonical keypoints, head pose and expression deformation of the input image
"""

import os
import pathlib
from torch import nn
import torch

from .convnextv2 import convnextv2_tiny
from .util import filter_state_dict

model_dict = {
    'convnextv2_tiny': convnextv2_tiny,
}


def torch_load_compat(ckpt_path, map_location=None, weights_only=None):
    load_kwargs = {}
    if map_location is not None:
        load_kwargs['map_location'] = map_location

    original_posix_path = None
    if os.name == 'nt':
        original_posix_path = pathlib.PosixPath
        pathlib.PosixPath = pathlib.PurePosixPath

    try:
        if weights_only is not None:
            try:
                return torch.load(ckpt_path, weights_only=weights_only, **load_kwargs)
            except TypeError:
                pass

        return torch.load(ckpt_path, **load_kwargs)
    finally:
        if original_posix_path is not None:
            pathlib.PosixPath = original_posix_path


class MotionExtractor(nn.Module):
    def __init__(self, **kwargs):
        super(MotionExtractor, self).__init__()

        # default is convnextv2_base
        backbone = kwargs.get('backbone', 'convnextv2_tiny')
        self.detector = model_dict.get(backbone)(**kwargs)

    def load_pretrained(self, init_path: str):
        if init_path not in (None, ''):
            checkpoint = torch_load_compat(
                init_path,
                map_location=lambda storage, loc: storage,
                weights_only=False
            )
            state_dict = checkpoint['model']
            state_dict = filter_state_dict(state_dict, remove_name='head')
            ret = self.detector.load_state_dict(state_dict, strict=False)
            print(f'Load pretrained model from {init_path}, ret: {ret}')

    def forward(self, x):
        out = self.detector(x)
        return out
