# borrow from optimization https://github.com/wangsen1312/joints2smpl

from __future__ import division, print_function

import argparse
import os
import random
import shutil
import sys
from os import listdir, walk
from os.path import isfile, join
from pathlib import Path

import h5py
import joblib
import natsort
import numpy as np
import smplx
import torch
import trimesh

from mGPT.data.transforms.joints2rots import config
from mGPT.data.transforms.joints2rots.smplify import SMPLify3D
from mGPT.utils.joints import mmm_to_smplh_scaling_factor
from mGPT.utils.temos_utils import subsample
from scripts.plys2npy import plys2npy

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# parsing argmument
parser = argparse.ArgumentParser()
parser.add_argument("--batchSize",
                    type=int,
                    default=1,
                    help="input batch size")
parser.add_argument(
    "--num_smplify_iters",
    type=int,
    default=100,
    help="num of smplify iters"  # 100
)
parser.add_argument("--cuda", type=bool, default=True, help="enables cuda")
parser.add_argument("--gpu_ids", type=int, default=0, help="choose gpu ids")
parser.add_argument("--num_joints", type=int, default=22, help="joint number")
parser.add_argument("--joint_category",
                    type=str,
                    default="AMASS",
                    help="use correspondence")
parser.add_argument("--fix_foot",
                    type=str,
                    default="False",
                    help="fix foot or not")
parser.add_argument(
    "--data_folder",
    type=str,
    default="",  # ./demo/demo_data/
    help="data in the folder",
)
parser.add_argument(
    "--save_folder",
    type=str,
    default=None,
    # default="./../TMOSTData/demo/",
    help="results save folder",
)
parser.add_argument("--dir", type=str, default=None, help="folder use")
parser.add_argument("--files",
                    type=str,
                    default="test_motion.npy",
                    help="files use")
opt = parser.parse_args()
print(opt)

# ---load predefined something
device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")
print(config.SMPL_MODEL_DIR)
# smplmodel = smplx.create(config.SMPL_MODEL_DIR,
#                          model_type="smplh", gender="neutral", ext="npz",
#                          batch_size=opt.batchSize).to(device)
smplmodel = smplx.create(
    config.SMPL_MODEL_DIR,
    model_type="smpl",
    gender="neutral",
    ext="pkl",
    batch_size=opt.batchSize,
).to(device)

# ## --- load the mean pose as original ----
smpl_mean_file = config.SMPL_MEAN_FILE

file = h5py.File(smpl_mean_file, "r")
init_mean_pose = (torch.from_numpy(
    file["pose"][:]).unsqueeze(0).float().repeat(opt.batchSize, 1).to(device))
init_mean_shape = (torch.from_numpy(
    file["shape"][:]).unsqueeze(0).float().repeat(opt.batchSize, 1).to(device))
cam_trans_zero = torch.Tensor([0.0, 0.0, 0.0]).unsqueeze(0).to(device)
#
pred_pose = torch.zeros(opt.batchSize, 72).to(device)
pred_betas = torch.zeros(opt.batchSize, 10).to(device)
pred_cam_t = torch.zeros(opt.batchSize, 3).to(device)
keypoints_3d = torch.zeros(opt.batchSize, opt.num_joints, 3).to(device)

# # #-------------initialize SMPLify
smplify = SMPLify3D(
    smplxmodel=smplmodel,
    batch_size=opt.batchSize,
    joints_category=opt.joint_category,
    num_iters=opt.num_smplify_iters,
    device=device,
)
print("initialize SMPLify3D done!")

paths = []
if opt.dir:
    output_dir = Path(opt.dir)
    # file_list = os.listdir(cfg.RENDER.DIR)
    # random begin for parallel
    file_list = natsort.natsorted(os.listdir(opt.dir))
    begin_id = random.randrange(0, len(file_list))
    file_list = file_list[begin_id:] + file_list[:begin_id]
    for item in file_list:
        if item.endswith(".npy"):
            paths.append(os.path.join(opt.dir, item))
elif opt.files:
    paths.append(opt.files)

print(f"begin to render {len(paths)} npy files!")

# if opt.save_folder is None:
#     save_folder = os.path.pardir(opt.dir) + "results_smplfitting"
#     if not os.path.isdir(save_folder):
#         os.makedirs(save_folder, exist_ok=True)

if not os.path.isdir(opt.save_folder):
    os.makedirs(opt.save_folder, exist_ok=True)

for path in paths:
    dir_save = os.path.join(opt.save_folder, "results_smplfitting",
                            "SMPLFit_" + os.path.basename(path)[:-4])

    if os.path.exists(path[:-4] + "_mesh.npy"):
        print(f"npy is fitted {path[:-4]}_mesh.npy")
        # check_file = ""
        # try:
        #     data = np.load(path)
        # except:
        continue

    # if os.path.exists(dir_save):
    #     print(f"npy is fitted or under fitting {dir_save}")
    #     continue

    data = np.load(path)
    if len(data.shape) > 3:
        data = data[0]

    # check input joint or meshes
    if data.shape[1] > 1000:
        print("npy is a mesh now {dir_save}")
        continue

    print(f"begin rendering {dir_save}")

    if not os.path.isdir(dir_save):
        os.makedirs(dir_save, exist_ok=True)

    if opt.num_joints == 22:
        #  humanml3d amass
        frames = subsample(len(data), last_framerate=12.5, new_framerate=12.5)
        data = data[frames, ...]
    elif opt.num_joints == 21:
        # kit
        # purename = os.path.splitext(opt.files)[0]
        # data = np.load(opt.data_folder + "/" + purename + ".npy")
        # downsampling to
        frames = subsample(len(data), last_framerate=100, new_framerate=12.5)
        data = data[frames, ...]
        # Convert mmm joints for visualization
        # into smpl-h "scale" and axis
        # data = data.copy()[..., [2, 0, 1]] * mmm_to_smplh_scaling_factor
        data = data.copy() * mmm_to_smplh_scaling_factor

    # run the whole seqs
    num_seqs = data.shape[0]

    pred_pose_prev = torch.zeros(opt.batchSize, 72).to(device)
    pred_betas_prev = torch.zeros(opt.batchSize, 10).to(device)
    pred_cam_t_prev = torch.zeros(opt.batchSize, 3).to(device)
    keypoints_3d_prev = torch.zeros(opt.batchSize, opt.num_joints,
                                    3).to(device)

    for idx in range(num_seqs):
        print(f"computing frame {idx}")

        ply_path = dir_save + "/" + "motion_%04d" % idx + ".ply"
        if os.path.exists(ply_path[:-4] + ".pkl"):
            print(f"this frame is fitted {ply_path}")
            continue

        joints3d = data[idx]  # *1.2 #scale problem [check first]
        keypoints_3d[0, :, :] = torch.Tensor(joints3d).to(device).float()

        if idx == 0:
            pred_betas[0, :] = init_mean_shape
            pred_pose[0, :] = init_mean_pose
            pred_cam_t[0, :] = cam_trans_zero
        else:
            # ToDo-use previous results rather than loading
            data_param = joblib.load(dir_save + "/" + "motion_%04d" %
                                     (idx - 1) + ".pkl")
            pred_betas[0, :] = torch.from_numpy(
                data_param["beta"]).unsqueeze(0).float()
            pred_pose[0, :] = torch.from_numpy(
                data_param["pose"]).unsqueeze(0).float()
            pred_cam_t[0, :] = torch.from_numpy(
                data_param["cam"]).unsqueeze(0).float()

        if opt.joint_category == "AMASS":
            confidence_input = torch.ones(opt.num_joints)
            # make sure the foot and ankle
            if opt.fix_foot == True:
                confidence_input[7] = 1.5
                confidence_input[8] = 1.5
                confidence_input[10] = 1.5
                confidence_input[11] = 1.5
        elif opt.joint_category == "MMM":
            confidence_input = torch.ones(opt.num_joints)
        else:
            print("Such category not settle down!")

        # ----- from initial to fitting -------
        (
            new_opt_vertices,
            new_opt_joints,
            new_opt_pose,
            new_opt_betas,
            new_opt_cam_t,
            new_opt_joint_loss,
        ) = smplify(
            pred_pose.detach(),
            pred_betas.detach(),
            pred_cam_t.detach(),
            keypoints_3d,
            conf_3d=confidence_input.to(device),
            # seq_ind=idx,
        )

        # # -- save the results to ply---
        outputp = smplmodel(
            betas=new_opt_betas,
            global_orient=new_opt_pose[:, :3],
            body_pose=new_opt_pose[:, 3:],
            transl=new_opt_cam_t,
            return_verts=True,
        )

        # gt debuggin
        if False:
            mesh_p = trimesh.Trimesh(
                vertices=keypoints_3d.detach().cpu().numpy().squeeze(),
                process=False)
            mesh_p.export(dir_save + "/" + "%04d" % idx + "_gt.ply")

        mesh_p = trimesh.Trimesh(
            vertices=outputp.vertices.detach().cpu().numpy().squeeze(),
            faces=smplmodel.faces,
            process=False,
        )
        mesh_p.export(ply_path)
        print("Output: " + ply_path)

        # save the pkl
        param = {}
        param["beta"] = new_opt_betas.detach().cpu().numpy()
        param["pose"] = new_opt_pose.detach().cpu().numpy()
        param["cam"] = new_opt_cam_t.detach().cpu().numpy()
        joblib.dump(param,
                    dir_save + "/" + "motion_%04d" % idx + ".pkl",
                    compress=3)
        print("Output: " + dir_save + "/" + "motion_%04d" % idx + ".pkl")

    print("merge ply to npy for mesh rendering")
    plys2npy(dir_save, os.path.dirname(path))

# # rendering
# if True:
#     from tmost.utils.demo_utils import render_batch
#     # render_batch(opt.dir, mode="sequence")  # sequence
#     render_batch(opt.dir, mode="video")
