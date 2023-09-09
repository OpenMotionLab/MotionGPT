import os
import argparse
import numpy as np
from scripts.hybrik_loc2rot import HybrIKJointsToRotmat
from scripts.pyrender import SMPLRender
import cv2
from scipy.spatial.transform import Rotation as RRR

parser = argparse.ArgumentParser(
    description='Render a SMPL video by a j3ds npy file.')
parser.add_argument('--input', type=str, default='', help='the npy file path')
parser.add_argument('--render',
                    type=int,
                    default=1,
                    help='render the video if 1')
args = parser.parse_args()

input_path = args.input
output_npy_path = args.input.replace('.npy', '_pose.npy')
data = np.load(input_path)
data = data - data[0, 0]
pose_generator = HybrIKJointsToRotmat()
pose = pose_generator(data)
pose = np.concatenate(
    [pose, np.stack([np.stack([np.eye(3)] * pose.shape[0], 0)] * 2, 1)], 1)
np.save(output_npy_path, pose)
shape = [768, 768]
if args.render:
    render = SMPLRender()
    output_mp4_path = args.input.replace('.npy', '_smpl.mp4')
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    size = (shape[1], shape[0])
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    videoWriter = cv2.VideoWriter(output_mp4_path, fourcc, fps, size)
    r = RRR.from_rotvec(np.array([np.pi, 0.0, 0.0]))
    pose[:, 0] = np.matmul(r.as_matrix().reshape(1, 3, 3), pose[:, 0])
    for i in range(data.shape[0]):
        img = np.zeros([shape[0], shape[1], 3])
        aroot = data[[i], 0] + np.array([[0.0, 0.0, 30.0]])
        aroot[:, 1] = -aroot[:, 1]
        params = dict(pred_shape=np.zeros([1, 10]),
                      pred_root=aroot,
                      pred_pose=pose[[i]])
        renderImg = render.render(img.copy(), params)
        renderImg = (renderImg * 255).astype(np.uint8)
        videoWriter.write(renderImg)
    videoWriter.release()
