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
from torch import Tensor

from .base import Datastruct, dataclass, Transform
from ..tools import collate_tensor_with_padding

from .joints2jfeats import Joints2Jfeats


class XYZTransform(Transform):
    def __init__(self, joints2jfeats: Joints2Jfeats, **kwargs):
        self.joints2jfeats = joints2jfeats

    def Datastruct(self, **kwargs):
        return XYZDatastruct(_joints2jfeats=self.joints2jfeats,
                             transforms=self,
                             **kwargs)

    def __repr__(self):
        return "XYZTransform()"


@dataclass
class XYZDatastruct(Datastruct):
    transforms: XYZTransform
    _joints2jfeats: Joints2Jfeats

    features: Optional[Tensor] = None
    joints_: Optional[Tensor] = None
    jfeats_: Optional[Tensor] = None

    def __post_init__(self):
        self.datakeys = ["features", "joints_", "jfeats_"]
        # starting point
        if self.features is not None and self.jfeats_ is None:
            self.jfeats_ = self.features

    @property
    def joints(self):
        # Cached value
        if self.joints_ is not None:
            return self.joints_

        # self.jfeats_ should be defined
        assert self.jfeats_ is not None

        self._joints2jfeats.to(self.jfeats.device)
        self.joints_ = self._joints2jfeats.inverse(self.jfeats)
        return self.joints_

    @property
    def jfeats(self):
        # Cached value
        if self.jfeats_ is not None:
            return self.jfeats_

        # self.joints_ should be defined
        assert self.joints_ is not None

        self._joints2jfeats.to(self.joints.device)
        self.jfeats_ = self._joints2jfeats(self.joints)
        return self.jfeats_

    def __len__(self):
        return len(self.jfeats)
