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
import smplx

from .base import Datastruct, dataclass, Transform

from .rots2rfeats import Rots2Rfeats
from .rots2joints import Rots2Joints
from .joints2jfeats import Joints2Jfeats


class SMPLTransform(Transform):
    def __init__(self, rots2rfeats: Rots2Rfeats,
                 rots2joints: Rots2Joints,
                 joints2jfeats: Joints2Jfeats,
                 **kwargs):
        self.rots2rfeats = rots2rfeats
        self.rots2joints = rots2joints
        self.joints2jfeats = joints2jfeats

    def Datastruct(self, **kwargs):
        return SMPLDatastruct(_rots2rfeats=self.rots2rfeats,
                              _rots2joints=self.rots2joints,
                              _joints2jfeats=self.joints2jfeats,
                              transforms=self,
                              **kwargs)

    def __repr__(self):
        return "SMPLTransform()"


class RotIdentityTransform(Transform):
    def __init__(self, **kwargs):
        return

    def Datastruct(self, **kwargs):
        return RotTransDatastruct(**kwargs)

    def __repr__(self):
        return "RotIdentityTransform()"


@dataclass
class RotTransDatastruct(Datastruct):
    rots: Tensor
    trans: Tensor

    transforms: RotIdentityTransform = RotIdentityTransform()

    def __post_init__(self):
        self.datakeys = ["rots", "trans"]

    def __len__(self):
        return len(self.rots)


@dataclass
class SMPLDatastruct(Datastruct):
    transforms: SMPLTransform
    _rots2rfeats: Rots2Rfeats
    _rots2joints: Rots2Joints
    _joints2jfeats: Joints2Jfeats

    features: Optional[Tensor] = None
    rots_: Optional[RotTransDatastruct] = None
    rfeats_: Optional[Tensor] = None
    joints_: Optional[Tensor] = None
    jfeats_: Optional[Tensor] = None
    vertices_: Optional[Tensor] = None

    def __post_init__(self):
        self.datakeys = ['features', 'rots_', 'rfeats_',
                         'joints_', 'jfeats_', 'vertices_']
        # starting point
        if self.features is not None and self.rfeats_ is None:
            self.rfeats_ = self.features

    @property
    def rots(self):
        # Cached value
        if self.rots_ is not None:
            return self.rots_

        # self.rfeats_ should be defined
        assert self.rfeats_ is not None

        self._rots2rfeats.to(self.rfeats.device)
        self.rots_ = self._rots2rfeats.inverse(self.rfeats)
        return self.rots_

    @property
    def rfeats(self):
        # Cached value
        if self.rfeats_ is not None:
            return self.rfeats_

        # self.rots_ should be defined
        assert self.rots_ is not None

        self._rots2rfeats.to(self.rots.device)
        self.rfeats_ = self._rots2rfeats(self.rots)
        return self.rfeats_

    @property
    def joints(self):
        # Cached value
        if self.joints_ is not None:
            return self.joints_

        self._rots2joints.to(self.rots.device)
        self.joints_ = self._rots2joints(self.rots)
        return self.joints_

    @property
    def jfeats(self):
        # Cached value
        if self.jfeats_ is not None:
            return self.jfeats_

        self._joints2jfeats.to(self.joints.device)
        self.jfeats_ = self._joints2jfeats(self.joints)
        return self.jfeats_
    
    @property
    def vertices(self):
        # Cached value
        if self.vertices_ is not None:
            return self.vertices_

        self._rots2joints.to(self.rots.device)
        self.vertices_ = self._rots2joints(self.rots, jointstype="vertices")
        return self.vertices_
    
    def __len__(self):
        return len(self.rfeats)


def get_body_model(model_type, gender, batch_size, device='cpu', ext='pkl'):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    mtype = model_type.upper()
    if gender != 'neutral':
        if not isinstance(gender, str):
            gender = str(gender.astype(str)).upper()
        else:
            gender = gender.upper()
    else:
        gender = gender.upper()
        ext = 'npz'
    body_model_path = f'data/smpl_models/{model_type}/{mtype}_{gender}.{ext}'

    body_model = smplx.create(body_model_path, model_type=type,
                              gender=gender, ext=ext,
                              use_pca=False,
                              num_pca_comps=12,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=batch_size)
    
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model

