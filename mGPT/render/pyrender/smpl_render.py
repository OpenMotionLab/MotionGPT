import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import cv2

import matplotlib.pyplot as plt
import glob
import pickle
import pyrender
import trimesh
from smplx import SMPL as _SMPL
from smplx.utils import SMPLOutput as ModelOutput
from scipy.spatial.transform.rotation import Rotation as RRR

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        # joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        # J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        # self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        # self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        # extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)        #Additional 9 joints #Check doc/J_regressor_extra.png
        # joints = torch.cat([smpl_output.joints, extra_joints], dim=1)               #[N, 24 + 21, 3]  + [N, 9, 3]
        # joints = joints[:, self.joint_map, :]
        joints = smpl_output.joints
        output = ModelOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=5000, img_res=(224,224), faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res[0],
                                       viewport_height=img_res[1],
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = [img_res[0] // 2, img_res[1] // 2]
        self.faces = faces
    def __call__(self, vertices, camera_translation, image):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=(0.8, 0.3, 0.3, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=(1.,1.,1.),ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(camera, pose=camera_pose)


        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img

class SMPLRender():
    def __init__(self, SMPL_MODEL_DIR):
        self.device = 'cuda'
        self.smpl = SMPL(SMPL_MODEL_DIR,
                             batch_size=1,
                             create_transl=False).to(self.device)

        self.focal_length = 5000

    def render(self, image, smpl_param, is_headroot=False):
        pose = smpl_param['pred_pose']
        if pose.size==72:
            pose = pose.reshape(-1,3)
            pose = RRR.from_rotvec(pose).as_matrix()
            pose = pose.reshape(1,24,3,3)
        pred_betas = torch.from_numpy(smpl_param['pred_shape'].reshape(1, 10).astype(np.float32)).to(self.device)
        pred_rotmat = torch.from_numpy(pose.astype(np.float32)).to(self.device)
        pred_camera_t = smpl_param['pred_root'].reshape(1, 3).astype(np.float32)
        smpl_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:, 1:],
                           global_orient=pred_rotmat[:, 0].unsqueeze(1), pose2rot=False)


        vertices = smpl_output.vertices[0].detach().cpu().numpy()
        pred_camera_t = pred_camera_t[0]

        if is_headroot:
            pred_camera_t = pred_camera_t - smpl_output.joints[0,12].detach().cpu().numpy()

        renderer = Renderer(focal_length=self.focal_length,
                                 img_res=(image.shape[1], image.shape[0]), faces=self.smpl.faces)

        renderImg = renderer(vertices, pred_camera_t.copy(), image / 255.0)
        renderer.renderer.delete()
        return renderImg
