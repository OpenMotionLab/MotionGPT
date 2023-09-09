import numpy as np

SMPL_BODY_BONES = [-0.0018, -0.2233, 0.0282, 0.0695, -0.0914, -0.0068, -0.0677, -0.0905, -0.0043,
                   -0.0025, 0.1090, -0.0267, 0.0343, -0.3752, -0.0045, -0.0383, -0.3826, -0.0089,
                   0.0055, 0.1352, 0.0011, -0.0136, -0.3980, -0.0437, 0.0158, -0.3984, -0.0423,
                   0.0015, 0.0529, 0.0254, 0.0264, -0.0558, 0.1193, -0.0254, -0.0481, 0.1233,
                   -0.0028, 0.2139, -0.0429, 0.0788, 0.1217, -0.0341, -0.0818, 0.1188, -0.0386,
                   0.0052, 0.0650, 0.0513, 0.0910, 0.0305, -0.0089, -0.0960, 0.0326, -0.0091,
                   0.2596, -0.0128, -0.0275, -0.2537, -0.0133, -0.0214, 0.2492, 0.0090, -0.0012,
                   -0.2553, 0.0078, -0.0056, 0.0840, -0.0082, -0.0149, -0.0846, -0.0061, -0.0103]


class HybrIKJointsToRotmat:
    def __init__(self):
        self.naive_hybrik = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                             0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        self.num_nodes = 22
        self.parents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        self.child = [-1, 4, 5, 6, 7, 8, 9, 10, 11, -1, -2, -2, 15,
                      16, 17, -2, 18, 19, 20, 21, -2, -2]
        self.bones = np.reshape(np.array(SMPL_BODY_BONES), [24, 3])[:self.num_nodes]

    def multi_child_rot(self, t, p,
                        pose_global_parent):
        """
        t: B x 3 x child_num
        p: B x 3 x child_num
        pose_global_parent: B x 3 x 3
        """
        m = np.matmul(t, np.transpose(np.matmul(np.linalg.inv(pose_global_parent), p), [0, 2, 1]))
        u, s, vt = np.linalg.svd(m)
        r = np.matmul(np.transpose(vt, [0, 2, 1]), np.transpose(u, [0, 2, 1]))
        err_det_mask = (np.linalg.det(r) < 0.0).reshape(-1, 1, 1)
        id_fix = np.reshape(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]),
                            [1, 3, 3])
        r_fix = np.matmul(np.transpose(vt, [0, 2, 1]),
                          np.matmul(id_fix,
                                    np.transpose(u, [0, 2, 1])))
        r = r * (1.0 - err_det_mask) + r_fix * err_det_mask
        return r, np.matmul(pose_global_parent, r)

    def single_child_rot(self, t, p, pose_global_parent, twist=None):
        """
        t: B x 3 x 1
        p: B x 3 x 1
        pose_global_parent: B x 3 x 3
        twist: B x 2 if given, default to None
        """
        p_rot = np.matmul(np.linalg.inv(pose_global_parent), p)
        cross = np.cross(t, p_rot, axisa=1, axisb=1, axisc=1)
        sina = np.linalg.norm(cross, axis=1, keepdims=True) / (np.linalg.norm(t, axis=1, keepdims=True) *
                                                               np.linalg.norm(p_rot, axis=1, keepdims=True))
        cross = cross / np.linalg.norm(cross, axis=1, keepdims=True)
        cosa = np.sum(t * p_rot, axis=1, keepdims=True) / (np.linalg.norm(t, axis=1, keepdims=True) *
                                                           np.linalg.norm(p_rot, axis=1, keepdims=True))
        sina = np.reshape(sina, [-1, 1, 1])
        cosa = np.reshape(cosa, [-1, 1, 1])
        skew_sym_t = np.stack([0.0 * cross[:, 0], -cross[:, 2], cross[:, 1],
                               cross[:, 2], 0.0 * cross[:, 0], -cross[:, 0],
                               -cross[:, 1], cross[:, 0], 0.0 * cross[:, 0]], 1)
        skew_sym_t = np.reshape(skew_sym_t, [-1, 3, 3])
        dsw_rotmat = np.reshape(np.eye(3), [1, 3, 3]
                                ) + sina * skew_sym_t + (1.0 - cosa) * np.matmul(skew_sym_t,
                                                                                 skew_sym_t)
        if twist is not None:
            skew_sym_t = np.stack([0.0 * t[:, 0], -t[:, 2], t[:, 1],
                                   t[:, 2], 0.0 * t[:, 0], -t[:, 0],
                                   -t[:, 1], t[:, 0], 0.0 * t[:, 0]], 1)
            skew_sym_t = np.reshape(skew_sym_t, [-1, 3, 3])
            sina = np.reshape(twist[:, 1], [-1, 1, 1])
            cosa = np.reshape(twist[:, 0], [-1, 1, 1])
            dtw_rotmat = np.reshape(np.eye(3), [1, 3, 3]
                                    ) + sina * skew_sym_t + (1.0 - cosa) * np.matmul(skew_sym_t,
                                                                                     skew_sym_t)
            dsw_rotmat = np.matmul(dsw_rotmat, dtw_rotmat)
        return dsw_rotmat, np.matmul(pose_global_parent, dsw_rotmat)

    def __call__(self, joints, twist=None):
        """
        joints: B x N x 3
        twist: B x N x 2 if given, default to None
        """
        expand_dim = False
        if len(joints.shape) == 2:
            expand_dim = True
            joints = np.expand_dims(joints, 0)
            if twist is not None:
                twist = np.expand_dims(twist, 0)
        assert (len(joints.shape) == 3)
        batch_size = np.shape(joints)[0]
        joints_rel = joints - joints[:, self.parents]
        joints_hybrik = 0.0 * joints_rel
        pose_global = np.zeros([batch_size, self.num_nodes, 3, 3])
        pose = np.zeros([batch_size, self.num_nodes, 3, 3])
        for i in range(self.num_nodes):
            if i == 0:
                joints_hybrik[:, 0] = joints[:, 0]
            else:
                joints_hybrik[:, i] = np.matmul(pose_global[:, self.parents[i]],
                                                np.reshape(self.bones[i], [1, 3, 1])).reshape(-1, 3) + \
                                      joints_hybrik[:, self.parents[i]]
            if self.child[i] == -2:
                pose[:, i] = pose[:, i] + np.eye(3).reshape(1, 3, 3)
                pose_global[:, i] = pose_global[:, self.parents[i]]
                continue
            if i == 0:
                r, rg = self.multi_child_rot(np.transpose(self.bones[[1, 2, 3]].reshape(1, 3, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [1, 2, 3]], [0, 2, 1]),
                                             np.eye(3).reshape(1, 3, 3))

            elif i == 9:
                r, rg = self.multi_child_rot(np.transpose(self.bones[[12, 13, 14]].reshape(1, 3, 3), [0, 2, 1]),
                                             np.transpose(joints_rel[:, [12, 13, 14]], [0, 2, 1]),
                                             pose_global[:, self.parents[9]])
            else:
                p = joints_rel[:, self.child[i]]
                if self.naive_hybrik[i] == 0:
                    p = joints[:, self.child[i]] - joints_hybrik[:, i]
                twi = None
                if twist is not None:
                    twi = twist[:, i]
                r, rg = self.single_child_rot(self.bones[self.child[i]].reshape(1, 3, 1),
                                              p.reshape(-1, 3, 1),
                                              pose_global[:, self.parents[i]],
                                              twi)
            pose[:, i] = r
            pose_global[:, i] = rg
        if expand_dim:
            pose = pose[0]
        return pose


if __name__ == "__main__":
    jts2rot_hybrik = HybrIKJointsToRotmat()
    joints = np.array(SMPL_BODY_BONES).reshape(1, 24, 3)[:, :22]
    parents = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    for i in range(1, 22):
        joints[:, i] = joints[:, i] + joints[:, parents[i]]
    pose = jts2rot_hybrik(joints)
    print(pose)
