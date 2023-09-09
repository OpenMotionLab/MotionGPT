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
from einops import rearrange
from torch import Tensor
from .tools import get_forward_direction, get_floor, gaussian_filter1d  # noqa
from mGPT.utils.geometry_tools import matrix_of_angles
from .base import Joints2Jfeats


class Rifke(Joints2Jfeats):

    def __init__(self,
                 jointstype: str = "mmm",
                 path: Optional[str] = None,
                 normalization: bool = False,
                 forward_filter: bool = False,
                 **kwargs) -> None:
        #
        # if jointstype != "mmm":
        #     print("This function assume that the root is the first index")
        #     raise NotImplementedError("This jointstype is not implemented.")

        super().__init__(path=path, normalization=normalization)
        self.jointstype = jointstype
        self.forward_filter = forward_filter

    def forward(self, joints: Tensor) -> Tensor:
        # Joints to rotation invariant poses (Holden et. al.)
        # Similar function than fke2rifke in Language2Pose repository
        # Adapted to pytorch
        # Put the origin center of the root joint instead of the ground projection
        poses = joints.clone()
        poses[..., 1] -= get_floor(poses, jointstype=self.jointstype)

        translation = poses[..., 0, :].clone()
        # Let the root have the Y translation --> gravity axis
        root_y = translation[..., 1]

        # Trajectory => Translation without gravity axis (Y)
        trajectory = translation[..., [0, 2]]

        # Delete the root joints of the poses
        poses = poses[..., 1:, :]

        # Remove the trajectory of the poses
        poses[..., [0, 2]] -= trajectory[..., None, :]

        # Compute the trajectory
        vel_trajectory = torch.diff(trajectory, dim=-2)
        # 0 for the first one => keep the dimentionality
        vel_trajectory = torch.cat(
            (0 * vel_trajectory[..., [0], :], vel_trajectory), dim=-2)

        # Compute the forward direction
        forward = get_forward_direction(poses, jointstype=self.jointstype)
        if self.forward_filter:
            # Smoothing to remove high frequencies
            forward = gaussian_filter1d(forward, 2)
            # normalize again to get real directions
            forward = torch.nn.functional.normalize(forward, dim=-1)
        # changed this also for New pytorch
        angles = torch.atan2(*(forward.transpose(0, -1))).transpose(0, -1)
        vel_angles = torch.diff(angles, dim=-1)
        # 0 for the first one => keep the dimentionality
        vel_angles = torch.cat((0 * vel_angles[..., [0]], vel_angles), dim=-1)

        # Construct the inverse rotation matrix
        sin, cos = forward[..., 0], forward[..., 1]
        rotations_inv = matrix_of_angles(cos, sin, inv=True)

        # Rotate the poses
        poses_local = torch.einsum("...lj,...jk->...lk", poses[..., [0, 2]],
                                   rotations_inv)
        poses_local = torch.stack(
            (poses_local[..., 0], poses[..., 1], poses_local[..., 1]), axis=-1)

        # stack the xyz joints into feature vectors
        poses_features = rearrange(poses_local,
                                   "... joints xyz -> ... (joints xyz)")

        # Rotate the vel_trajectory
        vel_trajectory_local = torch.einsum("...j,...jk->...k", vel_trajectory,
                                            rotations_inv)

        # Stack things together
        features = torch.cat((root_y[..., None], poses_features,
                              vel_angles[..., None], vel_trajectory_local), -1)

        # Normalize if needed
        features = self.normalize(features)
        return features

    def inverse(self, features: Tensor) -> Tensor:
        features = self.unnormalize(features)
        root_y, poses_features, vel_angles, vel_trajectory_local = self.extract(
            features)

        # already have the good dimensionality
        angles = torch.cumsum(vel_angles, dim=-1)
        # First frame should be 0, but if infered it is better to ensure it
        angles = angles - angles[..., [0]]

        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)

        # Get back the poses
        poses_local = rearrange(poses_features,
                                "... (joints xyz) -> ... joints xyz",
                                xyz=3)

        # Rotate the poses
        poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 2]],
                             rotations)
        poses = torch.stack(
            (poses[..., 0], poses_local[..., 1], poses[..., 1]), axis=-1)

        # Rotate the vel_trajectory
        vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local,
                                      rotations)
        # Integrate the trajectory
        # Already have the good dimensionality
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # Add the root joints (which is still zero)
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)

        # put back the root joint y
        poses[..., 0, 1] = root_y

        # Add the trajectory globally
        poses[..., [0, 2]] += trajectory[..., None, :]
        return poses

    def extract(self, features: Tensor):
        root_y = features[..., 0]
        poses_features = features[..., 1:-3]
        vel_angles = features[..., -3]
        vel_trajectory_local = features[..., -2:]

        return root_y, poses_features, vel_angles, vel_trajectory_local
