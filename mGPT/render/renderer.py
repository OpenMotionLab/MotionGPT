"""
This script is borrowed from https://github.com/mkocabas/VIBE
 Adhere to their licence to use this script
 It has been modified
"""

import os
import math
import trimesh

import pyrender
import numpy as np
from pyrender.constants import RenderFlags


# os.environ['DISPLAY'] = ':0.0'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
SMPL_MODEL_DIR = "data/smpl_data/"


def get_smpl_faces():
    return np.load(os.path.join(SMPL_MODEL_DIR, "smplfaces.npy"))


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, background=None, resolution=(224, 224), bg_color=[0, 0, 0, 0.5], orig_img=False, wireframe=False, cam_pose=np.eye(4)):
        width, height = resolution
        self.background = np.zeros((height, width, 3))
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=0.5
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.4, 0.4, 0.4))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=4)
    

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=np.dot(cam_pose,light_pose).copy())

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=np.dot(cam_pose,light_pose).copy())

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=np.dot(cam_pose,light_pose).copy())

        """ok
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)
        """

        # light_pose[:3, 3] = [0, -2, 2]
        # [droite, hauteur, profondeur camera]
        """
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)
        """

    def render(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9],
               cam_pose=np.eye(4)):
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)
        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        # Rx = trimesh.transformations.rotation_matrix(math.radians(-90), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=100000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0, # 0.0 for no specular lighting
            # metallicFactor=0.7, # 0.0 for no specular lighting
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        cam_node = self.scene.add(camera, pose=cam_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        if rgb.shape[-1]==3:
            # Debug
            # 0 not distinguish alpha
            valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
            output_img = rgb * valid_mask + (1 - valid_mask) * img
        elif rgb.shape[-1]==4:
            # valid_mask = (rgb[:, :, -1] > 128)[:, :, np.newaxis]
            # output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img

            # # output alpha
            valid_mask = (rgb[:, :, -1] > 128)[:, :]
            output_img = np.copy(rgb)
            output_img[:, :, -1] *= valid_mask           
            # output_img = img
        else:
            raise ValueError(f"rgb shape {rgb.shape[-1]} is not correct!")
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image


def get_renderer(width, height, cam_pose):
    renderer = Renderer(resolution=(width, height),
                        bg_color=[1, 1, 1, 0.5],
                        orig_img=False,
                        wireframe=False,
                        cam_pose=cam_pose)
    return renderer
