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

import contextlib
from typing import Optional
import torch
from torch import Tensor
from mGPT.utils.joints import smplh_to_mmm_scaling_factor, smplh2mmm_indexes, get_root_idx
from mGPT.utils.easyconvert import rep_to_rep
from .base import Rots2Joints


def slice_or_none(data, cslice):
    if data is None:
        return data
    else:
        return data[cslice]


class SMPLX(Rots2Joints):
    def __init__(self,
                 path: str,
                 jointstype: str = "mmm",
                 input_pose_rep: str = "matrix",
                 batch_size: int = 512,
                 gender="neutral",
                 **kwargs) -> None:
        super().__init__(path=None, normalization=False)
        self.batch_size = batch_size
        self.input_pose_rep = input_pose_rep
        self.jointstype = jointstype
        self.training = False

        from smplx.body_models import SMPLXLayer
        import os
        # rel_p = path.split('/')
        # rel_p = rel_p[rel_p.index('data'):]
        # rel_p = '/'.join(rel_p)

        # Remove annoying print
        with contextlib.redirect_stdout(None):
            self.smplx = SMPLXLayer(path,
                                    ext="npz",
                                    gender=gender,
                                    batch_size=batch_size).eval()

        self.faces = self.smplx.faces
        for p in self.parameters():
            p.requires_grad = False

    def train(self, *args, **kwargs):
        return self

    def forward(self,
                smpl_data: dict,
                jointstype: Optional[str] = None,
                input_pose_rep: Optional[str] = None,
                batch_size: Optional[int] = None) -> Tensor:

        # Take values from init if not specified there
        jointstype = self.jointstype if jointstype is None else jointstype
        batch_size = self.batch_size if batch_size is None else batch_size
        input_pose_rep = self.input_pose_rep if input_pose_rep is None else input_pose_rep

        poses = smpl_data.rots
        trans = smpl_data.trans

        from functools import reduce
        import operator
        save_shape_bs_len = poses.shape[:-3]
        nposes = reduce(operator.mul, save_shape_bs_len, 1)

        
        matrix_poses = rep_to_rep(self.input_pose_rep, input_pose_rep, poses)

        # Reshaping
        matrix_poses = matrix_poses.reshape((nposes, *matrix_poses.shape[-3:]))

        global_orient = matrix_poses[:, 0]

        if trans is None:
            trans = torch.zeros((*save_shape_bs_len, 3),
                                dtype=poses.dtype,
                                device=poses.device)

        trans_all = trans.reshape((nposes, *trans.shape[-1:]))

        body_pose = matrix_poses[:, 1:22]

        if poses.shape[-3] == 55:
            nohands = False
            nofaces = False
        elif poses.shape[-3] == 52:
            nohands = False
            nofaces = True
        elif poses.shape[-3] == 22:
            nohands = True
            nofaces = True
        else:
            raise NotImplementedError("Could not parse the poses.")

        if nohands:
            left_hand_pose = None
            right_hand_pose = None
        else:
            left_hand_pose = matrix_poses[:, 25:40]
            right_hand_pose = matrix_poses[:, 40:55]

        if nofaces:
            jaw_pose = None
            leye_pose = None
            reye_pose = None
        else:
            jaw_pose = matrix_poses[:, 22:23]
            leye_pose = matrix_poses[:, 23:24]
            reye_pose = matrix_poses[:, 24:25]

        n = len(body_pose)
        outputs = []
        for chunk in range(int((n - 1) / batch_size) + 1):
            chunk_slice = slice(chunk * batch_size, (chunk + 1) * batch_size)
            smpl_output = self.smplx(
                global_orient=slice_or_none(global_orient, chunk_slice),
                body_pose=slice_or_none(body_pose, chunk_slice),
                left_hand_pose=slice_or_none(left_hand_pose, chunk_slice),
                right_hand_pose=slice_or_none(right_hand_pose, chunk_slice),
                jaw_pose=slice_or_none(jaw_pose, chunk_slice),
                leye_pose=slice_or_none(leye_pose, chunk_slice),
                reye_pose=slice_or_none(reye_pose, chunk_slice),
                transl=slice_or_none(trans_all, chunk_slice))

            if jointstype == "vertices":
                output_chunk = smpl_output.vertices
            else:
                joints = smpl_output.joints
                output_chunk = joints

            outputs.append(output_chunk)

        outputs = torch.cat(outputs)
        outputs = outputs.reshape((*save_shape_bs_len, *outputs.shape[1:]))

        # Change topology if needed
        outputs = smplx_to(jointstype, outputs, trans)

        return outputs

    def inverse(self, joints: Tensor) -> Tensor:
        raise NotImplementedError("Cannot inverse SMPLX layer.")


def smplx_to(jointstype, data, trans):

    if "mmm" in jointstype:
        indexes = smplh2mmm_indexes
        data = data[..., indexes, :]

        # make it compatible with mmm
        if jointstype == "mmm":
            data *= smplh_to_mmm_scaling_factor

        if jointstype == "smplmmm":
            pass
        elif jointstype in ["mmm", "mmmns"]:
            # swap axis
            data = data[..., [1, 2, 0]]
            # revert left and right
            data[..., 2] = -data[..., 2]

    elif jointstype == "smplnh":
        from mGPT.utils.joints import smplh2smplnh_indexes
        indexes = smplh2smplnh_indexes
        data = data[..., indexes, :]
    elif jointstype == "smplh":
        pass
    elif jointstype == "vertices":
        pass
    else:
        raise NotImplementedError(f"SMPLX to {jointstype} is not implemented.")

    if jointstype != "vertices":
        # shift the output in each batch
        # such that it is centered on the pelvis/root on the first frame
        root_joint_idx = get_root_idx(jointstype)
        shift = trans[..., 0, :] - data[..., 0, root_joint_idx, :]
        data += shift[..., None, None, :]

    return data
