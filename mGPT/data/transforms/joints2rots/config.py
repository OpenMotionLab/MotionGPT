import numpy as np
from mGPT.utils.joints import mmm_joints, smplh2mmm_indexes

# Map joints Name to SMPL joints idx
JOINT_MAP = {
    'MidHip': 0,
    'LHip': 1,
    'LKnee': 4,
    'LAnkle': 7,
    'LFoot': 10,
    'RHip': 2,
    'RKnee': 5,
    'RAnkle': 8,
    'RFoot': 11,
    'LShoulder': 16,
    'LElbow': 18,
    'LWrist': 20,
    'LHand': 22,
    'RShoulder': 17,
    'RElbow': 19,
    'RWrist': 21,
    'RHand': 23,
    'spine1': 3,
    'spine2': 6,
    'spine3': 9,
    'Neck': 12,
    'Head': 15,
    'LCollar': 13,
    'Rcollar': 14,
    'Nose': 24,
    'REye': 26,
    'LEye': 26,
    'REar': 27,
    'LEar': 28,
    'LHeel': 31,
    'RHeel': 34,
    'OP RShoulder': 17,
    'OP LShoulder': 16,
    'OP RHip': 2,
    'OP LHip': 1,
    'OP Neck': 12,
}

mmm2smpl_correspondence = {
    "root": "MidHip",
    "BP": "spine1",
    "BT": "spine3",
    "BLN": "Neck",
    "BUN": "Head",
    "LS": "LShoulder",
    "LE": "LElbow",
    "LW": "LWrist",
    "RS": "RShoulder",
    "RE": "RElbow",
    "RW": "RWrist",
    "LH": "LHip",
    "LK": "LKnee",
    "LA": "LAnkle",
    "LMrot": "LHeel",
    "LF": "LFoot",
    "RH": "RHip",
    "RK": "RKnee",
    "RA": "RAnkle",
    "RMrot": "RHeel",
    "RF": "RFoot"
}

full_smpl_idx = range(24)
key_smpl_idx = [0, 1, 4, 7, 2, 5, 8, 17, 19, 21, 16, 18, 20]

AMASS_JOINT_MAP = {
    'MidHip': 0,
    'LHip': 1,
    'LKnee': 4,
    'LAnkle': 7,
    'LFoot': 10,
    'RHip': 2,
    'RKnee': 5,
    'RAnkle': 8,
    'RFoot': 11,
    'LShoulder': 16,
    'LElbow': 18,
    'LWrist': 20,
    'RShoulder': 17,
    'RElbow': 19,
    'RWrist': 21,
    'spine1': 3,
    'spine2': 6,
    'spine3': 9,
    'Neck': 12,
    'Head': 15,
    'LCollar': 13,
    'Rcollar': 14,
}
amass_idx = range(22)
amass_smpl_idx = range(22)

# cal mmm in smpl index
smpl2mmm_correspondence = {
    val: key
    for key, val in mmm2smpl_correspondence.items()
}
smpl2mmm_indexes = [JOINT_MAP[mmm2smpl_correspondence[x]] for x in mmm_joints]

# cal mmm joints map
MMM_JOINT_MAP = {
    val: JOINT_MAP[val]
    for key, val in mmm2smpl_correspondence.items()
}

# mmm_idx = range(21)
# mmm_smpl_dix = smpl2mmm_indexes
# mmm_smpl_dix = smplh2mmm_indexes
# todo - configable
SMPL_MODEL_DIR = "/apdcephfs/share_1227775/shingxchen/AIMotion/TMOSTData/deps/smpl_models/"
GMM_MODEL_DIR = "/apdcephfs/share_1227775/shingxchen/AIMotion/TMOSTData/deps/smpl_models/"
SMPL_MEAN_FILE = "/apdcephfs/share_1227775/shingxchen/AIMotion/TMOSTData/deps/smpl_models/neutral_smpl_mean_params.h5"
# for collsion
Part_Seg_DIR = "/apdcephfs/share_1227775/shingxchen/AIMotion/TMOSTData/deps/smpl_models/smplx_parts_segm.pkl"
