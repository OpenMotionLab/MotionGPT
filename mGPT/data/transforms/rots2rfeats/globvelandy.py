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
from torch import Tensor
from einops import rearrange

from mGPT.utils.easyconvert import rep_to_rep, nfeats_of, to_matrix
import mGPT.utils.geometry_tools as geometry_tools

from .base import Rots2Rfeats


class Globalvelandy(Rots2Rfeats):
    def __init__(self,
                 path: Optional[str] = None,
                 normalization: bool = False,
                 pose_rep: str = "rot6d",
                 canonicalize: bool = False,
                 offset: bool = True,
                 **kwargs) -> None:
        super().__init__(path=path, normalization=normalization)

        self.canonicalize = canonicalize
        self.pose_rep = pose_rep
        self.nfeats = nfeats_of(pose_rep)
        self.offset = offset

    def forward(self, data, data_rep='matrix', first_frame=None) -> Tensor:

        poses, trans = data.rots, data.trans

        # extract the root gravity axis
        # for smpl it is the last coordinate
        root_y = trans[..., 2]
        trajectory = trans[..., [0, 1]]

        # Compute the difference of trajectory
        vel_trajectory = torch.diff(trajectory, dim=-2)

        # 0 for the first one => keep the dimentionality
        if first_frame is None:
            first_frame = 0 * vel_trajectory[..., [0], :]

        vel_trajectory = torch.cat((first_frame, vel_trajectory), dim=-2)

        # first normalize the data
        if self.canonicalize:

            matrix_poses = rep_to_rep(data_rep, 'matrix', poses)
            global_orient = matrix_poses[..., 0, :, :]

            # remove the rotation
            rot2d = rep_to_rep(data_rep, 'rotvec', poses[0, 0, ...])
            
            # Remove the fist rotation along the vertical axis
            rot2d[..., :2] = 0

            if self.offset:
                # add a bit more rotation
                rot2d[..., 2] += torch.pi / 2

            rot2d = rep_to_rep('rotvec', 'matrix', rot2d)
            
            # turn with the same amount all the rotations
            global_orient = torch.einsum("...kj,...kl->...jl", rot2d,
                                         global_orient)

            matrix_poses = torch.cat(
                (global_orient[..., None, :, :], matrix_poses[..., 1:, :, :]),
                dim=-3)

            poses = rep_to_rep('matrix', data_rep, matrix_poses)

            # Turn the trajectory as well
            vel_trajectory = torch.einsum("...kj,...lk->...lj",
                                          rot2d[..., :2, :2], vel_trajectory)

        poses = rep_to_rep(data_rep, self.pose_rep, poses)
        features = torch.cat(
            (root_y[..., None], vel_trajectory,
             rearrange(poses, "... joints rot -> ... (joints rot)")),
            dim=-1)
        features = self.normalize(features)

        return features

    def extract(self, features):
        root_y = features[..., 0]
        vel_trajectory = features[..., 1:3]
        poses_features = features[..., 3:]
        poses = rearrange(poses_features,
                          "... (joints rot) -> ... joints rot",
                          rot=self.nfeats)
        return root_y, vel_trajectory, poses

    def inverse(self, features, last_frame=None):
        features = self.unnormalize(features)
        root_y, vel_trajectory, poses = self.extract(features)

        # integrate the trajectory
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        if last_frame is None:
            pass
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # Get back the translation
        trans = torch.cat([trajectory, root_y[..., None]], dim=-1)
        matrix_poses = rep_to_rep(self.pose_rep, 'matrix',  poses)

        from ..smpl import RotTransDatastruct
        return RotTransDatastruct(rots=matrix_poses, trans=trans)
