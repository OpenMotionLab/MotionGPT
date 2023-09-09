import math
import os
import sys
import smplx
import bpy
import numpy as np

from .camera import Camera
from .floor import get_trajectory, plot_floor, show_traj
from .sampler import get_frameidx
from .scene import setup_scene  # noqa
from .tools import delete_objs, load_numpy_vertices_into_blender, style_detect
from .vertices import prepare_vertices
from mGPT.utils.joints import smplh_to_mmm_scaling_factor


def prune_begin_end(data, perc):
    to_remove = int(len(data) * perc)
    if to_remove == 0:
        return data
    return data[to_remove:-to_remove]


def render_current_frame(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(use_viewport=True, write_still=True)


def render(npydata,
           frames_folder,
           *,
           mode,
           model_path,
           faces_path,
           gt=False,
           exact_frame=None,
           num=8,
           downsample=True,
           canonicalize=True,
           always_on_floor=False,
           denoising=True,
           oldrender=True,
           res="high",
           init=True,
           accelerator='gpu',
           device=[0]):
    if init:
        # Setup the scene (lights / render engine / resolution etc)
        setup_scene(res=res,
                    denoising=denoising,
                    oldrender=oldrender,
                    accelerator=accelerator,
                    device=device)

    is_mesh, is_smplx, jointstype = style_detect(npydata)
    
    if not is_mesh:
        npydata = npydata * smplh_to_mmm_scaling_factor

    if is_smplx:
        smplx_model_male = smplx.create(model_path,
                                        model_type='smplx',
                                        gender='male',
                                        ext='npz',
                                        num_betas=10,
                                        flat_hand_mean=True,
                                        use_pca=False)
        faces_path = smplx_model_male.faces

    

    # Put everything in this folder
    if mode == "video":
        if always_on_floor:
            frames_folder += "_of"
        os.makedirs(frames_folder, exist_ok=True)
        # if it is a mesh, it is already downsampled
        if downsample and not is_mesh:
            npydata = npydata[::8]
    elif mode == "sequence":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}{ext}"

    elif mode == "frame":
        img_name, ext = os.path.splitext(frames_folder)
        if always_on_floor:
            img_name += "_of"
        img_path = f"{img_name}_{exact_frame}{ext}"

    # remove X% of begining and end
    # as it is almost always static
    # in this part
    if mode == "sequence":
        perc = 0.2
        npydata = prune_begin_end(npydata, perc)

    if is_mesh:
        from .meshes import Meshes
        data = Meshes(npydata,
                      gt=gt,
                      mode=mode,
                      faces_path=faces_path,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor,
                      is_smplx=is_smplx)
    else:
        from .joints import Joints
        data = Joints(npydata,
                      gt=gt,
                      mode=mode,
                      canonicalize=canonicalize,
                      always_on_floor=always_on_floor,
                      jointstype=jointstype)

    # Number of frames possible to render
    nframes = len(data)

    # Show the trajectory
    show_traj(data.trajectory)

    # Create a floor
    plot_floor(data.data, big_plane=False)

    # initialize the camera
    camera = Camera(first_root=data.get_root(0), mode=mode, is_mesh=is_mesh)

    frameidx = get_frameidx(mode=mode,
                            nframes=nframes,
                            exact_frame=exact_frame,
                            frames_to_keep=num)

    nframes_to_render = len(frameidx)

    # center the camera to the middle
    if mode == "sequence":
        camera.update(data.get_mean_root())

    imported_obj_names = []
    for index, frameidx in enumerate(frameidx):
        if mode == "sequence":
            frac = index / (nframes_to_render - 1)
            mat = data.get_sequence_mat(frac)
        else:
            mat = data.mat
            camera.update(data.get_root(frameidx))

        islast = index == (nframes_to_render - 1)

        objname = data.load_in_blender(frameidx, mat)
        name = f"{str(index).zfill(4)}"

        if mode == "video":
            path = os.path.join(frames_folder, f"frame_{name}.png")
        else:
            path = img_path

        if mode == "sequence":
            imported_obj_names.extend(objname)
        elif mode == "frame":
            camera.update(data.get_root(frameidx))

        if mode != "sequence" or islast:
            render_current_frame(path)
            delete_objs(objname)

    bpy.ops.wm.save_as_mainfile(filepath=frames_folder.replace('.png','.blend').replace('_frames','.blend'))

    # remove every object created
    delete_objs(imported_obj_names)
    delete_objs(["Plane", "myCurve", "Cylinder"])

    if mode == "video":
        return frames_folder
    else:
        return img_path
