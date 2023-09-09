from typing import Dict, List

import numpy as np
import torch
from torch import Tensor

import mGPT.utils.geometry_conver as geometry_conver


def lengths_to_mask(lengths: List[int],
                    device: torch.device,
                    max_len: int = None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def remove_padding(tensors, lengths):
    return [
        tensor[:tensor_length]
        for tensor, tensor_length in zip(tensors, lengths)
    ]


def nfeats_of(rottype):
    if rottype in ["rotvec", "axisangle"]:
        return 3
    elif rottype in ["rotquat", "quaternion"]:
        return 4
    elif rottype in ["rot6d", "6drot", "rotation6d"]:
        return 6
    elif rottype in ["rotmat"]:
        return 9
    else:
        return TypeError("This rotation type doesn't have features.")


def axis_angle_to(newtype, rotations):
    if newtype in ["matrix"]:
        rotations = geometry_conver.axis_angle_to_matrix(rotations)
        return rotations
    elif newtype in ["rotmat"]:
        rotations = geometry_conver.axis_angle_to_matrix(rotations)
        rotations = matrix_to("rotmat", rotations)
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry_conver.axis_angle_to_matrix(rotations)
        rotations = matrix_to("rot6d", rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = geometry_conver.axis_angle_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        return rotations
    else:
        raise NotImplementedError


def matrix_to(newtype, rotations):
    if newtype in ["matrix"]:
        return rotations
    if newtype in ["rotmat"]:
        rotations = rotations.reshape((*rotations.shape[:-2], 9))
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry_conver.matrix_to_rotation_6d(rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = geometry_conver.matrix_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        rotations = geometry_conver.matrix_to_axis_angle(rotations)
        return rotations
    else:
        raise NotImplementedError


def to_matrix(oldtype, rotations):
    if oldtype in ["matrix"]:
        return rotations
    if oldtype in ["rotmat"]:
        rotations = rotations.reshape((*rotations.shape[:-2], 3, 3))
        return rotations
    elif oldtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = geometry_conver.rotation_6d_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotquat", "quaternion"]:
        rotations = geometry_conver.quaternion_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotvec", "axisangle"]:
        rotations = geometry_conver.axis_angle_to_matrix(rotations)
        return rotations
    else:
        raise NotImplementedError


# TODO: use a real subsampler..
def subsample(num_frames, last_framerate, new_framerate):
    step = int(last_framerate / new_framerate)
    assert step >= 1
    frames = np.arange(0, num_frames, step)
    return frames


# TODO: use a real upsampler..
def upsample(motion, last_framerate, new_framerate):
    step = int(new_framerate / last_framerate)
    assert step >= 1

    # Alpha blending => interpolation
    alpha = np.linspace(0, 1, step + 1)
    last = np.einsum("l,...->l...", 1 - alpha, motion[:-1])
    new = np.einsum("l,...->l...", alpha, motion[1:])

    chuncks = (last + new)[:-1]
    output = np.concatenate(chuncks.swapaxes(1, 0))
    # Don't forget the last one
    output = np.concatenate((output, motion[[-1]]))
    return output


if __name__ == "__main__":
    motion = np.arange(105)
    submotion = motion[subsample(len(motion), 100.0, 12.5)]
    newmotion = upsample(submotion, 12.5, 100)

    print(newmotion)
