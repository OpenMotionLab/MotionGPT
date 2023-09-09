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

import torch
import torch.nn.functional as F

from mGPT.utils.joints import mmm_joints

# Get the indexes of particular body part SMPLH case
# Feet
# LM, RM = smplh_joints.index("left_ankle"), smplh_joints.index("right_ankle")
# LF, RF = smplh_joints.index("left_foot"), smplh_joints.index("right_foot")
# # Shoulders
# LS, RS = smplh_joints.index("left_shoulder"), smplh_joints.index("right_shoulder")
# # Hips
# LH, RH = smplh_joints.index("left_hip"), smplh_joints.index("right_hip")

# Get the indexes of particular body part
# Feet
LM, RM = mmm_joints.index("LMrot"), mmm_joints.index("RMrot")
LF, RF = mmm_joints.index("LF"), mmm_joints.index("RF")
# Shoulders
LS, RS = mmm_joints.index("LS"), mmm_joints.index("RS")
# Hips
LH, RH = mmm_joints.index("LH"), mmm_joints.index("RH")


def get_forward_direction(poses, jointstype="mmm"):
    # assert jointstype == 'mmm'
    across = poses[..., RH, :] - poses[..., LH, :] + poses[..., RS, :] - poses[
        ..., LS, :]
    forward = torch.stack((-across[..., 2], across[..., 0]), axis=-1)
    forward = torch.nn.functional.normalize(forward, dim=-1)
    return forward


def get_floor(poses, jointstype="mmm"):
    # assert jointstype == 'mmm'
    ndim = len(poses.shape)
    foot_heights = poses[..., (LM, LF, RM, RF), 1].min(-1).values
    floor_height = softmin(foot_heights, softness=0.5, dim=-1)
    # changed this thing Mathis version 1.11 pytorch
    return floor_height[(ndim - 2) * [None]].transpose(0, -1)


def softmax(x, softness=1.0, dim=None):
    maxi, mini = x.max(dim=dim).values, x.min(dim=dim).values
    return maxi + torch.log(softness + torch.exp(mini - maxi))


def softmin(x, softness=1.0, dim=0):
    return -softmax(-x, softness=softness, dim=dim)


def gaussian_filter1d(_inputs, sigma, truncate=4.0):
    # Code adapted/mixed from scipy library into pytorch
    # https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/ndimage/filters.py#L211
    # and gaussian kernel
    # https://github.com/scipy/scipy/blob/47bb6febaa10658c72962b9615d5d5aa2513fa3a/scipy/ndimage/filters.py#L179
    # Correspond to mode="nearest" and order = 0
    # But works batched
    if len(_inputs.shape) == 2:
        inputs = _inputs[None]
    else:
        inputs = _inputs

    sd = float(sigma)
    radius = int(truncate * sd + 0.5)
    sigma2 = sigma * sigma
    x = torch.arange(-radius,
                     radius + 1,
                     device=inputs.device,
                     dtype=inputs.dtype)
    phi_x = torch.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    # Conv1d weights
    groups = inputs.shape[-1]
    weights = torch.tile(phi_x, (groups, 1, 1))
    inputs = inputs.transpose(-1, -2)
    outputs = F.conv1d(inputs, weights, padding="same",
                       groups=groups).transpose(-1, -2)

    return outputs.reshape(_inputs.shape)
