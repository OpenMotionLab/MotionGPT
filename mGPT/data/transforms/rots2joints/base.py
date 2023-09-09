# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from typing import Optional

import torch
from torch import Tensor, nn
from pathlib import Path
import os
# import hydra

class Rots2Joints(nn.Module):
    def __init__(self, path: Optional[str] = None,
                 normalization: bool = False,
                 eps: float = 1e-12,
                 **kwargs) -> None:
        if normalization and path is None:
            raise TypeError("You should provide a path if normalization is on.")

        super().__init__()
        self.normalization = normalization
        self.eps = eps
        # workaround for cluster local/sync
        if path is not None:
            rel_p = path.split('/')
            rel_p = rel_p[rel_p.index('deps'):]
            rel_p = '/'.join(rel_p)
            # path = hydra.utils.get_original_cwd() + '/' + rel_p
        if normalization:
            mean_path = Path(path) / "mean.pt"
            std_path = Path(path) / "std.pt"
            self.register_buffer('mean', torch.load(mean_path))
            self.register_buffer('std', torch.load(std_path))

    def normalize(self, features: Tensor) -> Tensor:
        if self.normalization:
            features = (features - self.mean)/(self.std + self.eps)
        return features

    def unnormalize(self, features: Tensor) -> Tensor:
        if self.normalization:
            features = features * self.std + self.mean
        return features
