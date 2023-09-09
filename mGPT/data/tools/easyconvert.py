from .geometry import *

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
        rotations = axis_angle_to_matrix(rotations)
        return rotations
    elif newtype in ["rotmat"]:
        rotations = axis_angle_to_matrix(rotations)
        rotations = matrix_to("rotmat", rotations)
        return rotations
    elif newtype in ["rot6d", "6drot", "rotation6d"]:
        rotations = axis_angle_to_matrix(rotations)
        rotations = matrix_to("rot6d", rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = axis_angle_to_quaternion(rotations)
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
        rotations = matrix_to_rotation_6d(rotations)
        return rotations
    elif newtype in ["rotquat", "quaternion"]:
        rotations = matrix_to_quaternion(rotations)
        return rotations
    elif newtype in ["rotvec", "axisangle"]:
        rotations = matrix_to_axis_angle(rotations)
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
        rotations = rotation_6d_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotquat", "quaternion"]:
        rotations = quaternion_to_matrix(rotations)
        return rotations
    elif oldtype in ["rotvec", "axisangle"]:
        rotations = axis_angle_to_matrix(rotations)
        return rotations
    else:
        raise NotImplementedError
