mmm_joints = [
    "root",
    "BP",
    "BT",
    "BLN",
    "BUN",
    "LS",
    "LE",
    "LW",
    "RS",
    "RE",
    "RW",
    "LH",
    "LK",
    "LA",
    "LMrot",
    "LF",
    "RH",
    "RK",
    "RA",
    "RMrot",
    "RF",
]

humanml3d_joints = [
    "root",
    "RH",
    "LH",
    "BP",
    "RK",
    "LK",
    "BT",
    "RMrot",
    "LMrot",
    "BLN",
    "RF",
    "LF",
    "BMN",
    "RSI",
    "LSI",
    "BUN",
    "RS",
    "LS",
    "RE",
    "LE",
    "RW",
    "LW",
]

smplx_joints = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

smplxnh_joints = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

smplh_joints = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
]

smplnh_joints = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

mmm2smplh_correspondence = {
    "root": "pelvis",
    "BP": "spine1",
    "BT": "spine3",
    "BLN": "neck",
    "BUN": "head",
    "LS": "left_shoulder",
    "LE": "left_elbow",
    "LW": "left_wrist",
    "RS": "right_shoulder",
    "RE": "right_elbow",
    "RW": "right_wrist",
    "LH": "left_hip",
    "LK": "left_knee",
    "LA": "left_ankle",
    "LMrot": "left_heel",
    "LF": "left_foot",
    "RH": "right_hip",
    "RK": "right_knee",
    "RA": "right_ankle",
    "RMrot": "right_heel",
    "RF": "right_foot",
}

smplh2mmm_correspondence = {
    val: key
    for key, val in mmm2smplh_correspondence.items()
}
smplh2mmm_indexes = [
    smplh_joints.index(mmm2smplh_correspondence[x]) for x in mmm_joints
]

smplnh2smplh_correspondence = {key: key for key in smplnh_joints}
smplh2smplnh_correspondence = {
    val: key
    for key, val in smplnh2smplh_correspondence.items()
}

smplh2smplnh_indexes = [
    smplh_joints.index(smplnh2smplh_correspondence[x]) for x in smplnh_joints
]

mmm_kinematic_tree = [
    [0, 1, 2, 3, 4],  # body
    [3, 5, 6, 7],  # right arm
    [3, 8, 9, 10],  # left arm
    [0, 11, 12, 13, 14, 15],  # right leg
    [0, 16, 17, 18, 19, 20],
]  # left leg

humanml3d_kinematic_tree = [
    [0, 3, 6, 9, 12, 15],  # body
    [9, 14, 17, 19, 21],  # right arm
    [9, 13, 16, 18, 20],  # left arm
    [0, 2, 5, 8, 11],  # right leg
    [0, 1, 4, 7, 10],
]  # left leg

smplh_to_mmm_scaling_factor = 480 / 0.75
mmm_to_smplh_scaling_factor = 0.75 / 480

mmm_joints_info = {
    "root":
    mmm_joints.index("root"),
    "feet": [
        mmm_joints.index("LMrot"),
        mmm_joints.index("RMrot"),
        mmm_joints.index("LF"),
        mmm_joints.index("RF"),
    ],
    "shoulders": [mmm_joints.index("LS"),
                  mmm_joints.index("RS")],
    "hips": [mmm_joints.index("LH"),
             mmm_joints.index("RH")],
}

smplnh_joints_info = {
    "root":
    smplnh_joints.index("pelvis"),
    "feet": [
        smplnh_joints.index("left_ankle"),
        smplnh_joints.index("right_ankle"),
        smplnh_joints.index("left_foot"),
        smplnh_joints.index("right_foot"),
    ],
    "shoulders": [
        smplnh_joints.index("left_shoulder"),
        smplnh_joints.index("right_shoulder"),
    ],
    "hips":
    [smplnh_joints.index("left_hip"),
     smplnh_joints.index("right_hip")],
}

infos = {"mmm": mmm_joints_info, "smplnh": smplnh_joints_info}

smplh_indexes = {"mmm": smplh2mmm_indexes, "smplnh": smplh2smplnh_indexes}

root_joints = {
    "mmm": mmm_joints_info["root"],
    "mmmns": mmm_joints_info["root"],
    "smplmmm": mmm_joints_info["root"],
    "smplnh": smplnh_joints_info["root"],
    "smplh": smplh_joints.index("pelvis"),
}


def get_root_idx(joinstype):
    return root_joints[joinstype]


# def mmm2smpl(joints_mmm):
#     mmm2smplnh_indexes = []
#     for x in smplnh_joints:
#         if x in smplh2mmm_correspondence:
#             mmm2smplnh_indexes.append(mmm_joints.index(smplh2mmm_correspondence[x]))

#     spine2 = 0.5*(joints[mmm_joints.index("spine1")] + joints[mmm_joints.index("spine3")])

#     joints = joints_mmm[indexes]
#     return joints
