import numpy as np
import imageio
import os
import argparse
from tqdm import tqdm
from .renderer import get_renderer


def get_rotation(theta=np.pi / 3):
    import mGPT.utils.rotation_conversions as geometry
    import torch
    axis = torch.tensor([0, 1, 0], dtype=torch.float)
    axisangle = theta * axis
    matrix = geometry.axis_angle_to_matrix(axisangle)
    return matrix.numpy()


def render_video(meshes,
                 key,
                 action,
                 renderer,
                 savepath,
                 backgrounds,
                 cam_pose,
                 cams=(0.75, 0.75, 0, 0.10),
                 color=[0.11, 0.53, 0.8]):
    #  cams=(0.75, 0.75, 0, 0.10), color=[165.0/255,112/255,140/255]):
    # center the first frame
    if key not in ["real", "ntf", "side"]:
        w = int(key) / 6.0
        # purpole to green
        # color = w*np.array([0.9,102/255,120/255]) + (1-w)*np.array([0.11, 0.9, 0.11])
        # color = (1-w)*np.array([165.0/255,112/255,140/255]) + w*np.array([0.11, 0.8, 0.11])
        color = (1 - w) * np.array([0.75, 0.13, 0.7]) + w * np.array(
            [0.12, 0.7, 0.14])

    meshes = meshes - meshes[0].mean(axis=0)
    imgs = []
    idx = 0
    # for mesh in meshes:
    for mesh in tqdm(meshes, desc=f"Visualize {key}, action {action}"):
        # file_name = '3dpw_rot-90_glob_trimesh.ply' mesh_filename=file_name,
        # prepare background
        if len(backgrounds.shape) == 3:
            background = backgrounds
            cam = cams
        elif len(backgrounds.shape) == 4:
            background = backgrounds[idx]
            cam = cams[idx]
            idx += 1
        # prepare cams
        img = renderer.render(background,
                              mesh,
                              cam,
                              color=color,
                              cam_pose=cam_pose)
        imgs.append(img)
        # show(img)

    imgs = np.array(imgs)
    # masks = ~(imgs/255. > 0.96).all(-1)
    # coords = np.argwhere(masks.sum(axis=0))
    # y1, x1 = coords.min(axis=0)
    # y2, x2 = coords.max(axis=0)
    # writer = imageio.get_writer(savepath, fps=30)
    # for cimg in imgs[:, y1:y2, x1:x2]:
    #     writer.append_data(cimg)
    # writer.close()

    # from mld.utils.uicap_utils import write_rgba_seqs
    # write_rgba_seqs(imgs, savepath)

    writer = imageio.get_writer(savepath, fps=30)
    for cimg in imgs:
        writer.append_data(cimg)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    opt = parser.parse_args()
    filename = opt.filename
    savefolder = os.path.splitext(filename)[0]
    os.makedirs(savefolder, exist_ok=True)

    output = np.load(filename)

    if output.shape[0] == 3:
        visualization, generation, reconstruction = output
        output = {
            "visualization": visualization,
            "generation": generation,
            "reconstruction": reconstruction
        }
    else:
        # output = {f"generation_{key}": output[key] for key in range(2)} #  len(output))}
        # output = {f"generation_{key}": output[key] for key in range(len(output))}
        output = {
            f"generation_{key}": output[key]
            for key in range(len(output))
        }

    width = 1024
    height = 1024

    background = np.zeros((height, width, 3))
    renderer = get_renderer(width, height)

    # if duration mode, put back durations
    if output["generation_3"].shape[-1] == 100:
        output["generation_0"] = output["generation_0"][:, :, :, :40]
        output["generation_1"] = output["generation_1"][:, :, :, :60]
        output["generation_2"] = output["generation_2"][:, :, :, :80]
        output["generation_3"] = output["generation_3"][:, :, :, :100]
    elif output["generation_3"].shape[-1] == 160:
        print("160 mode")
        output["generation_0"] = output["generation_0"][:, :, :, :100]
        output["generation_1"] = output["generation_1"][:, :, :, :120]
        output["generation_2"] = output["generation_2"][:, :, :, :140]
        output["generation_3"] = output["generation_3"][:, :, :, :160]

    # if str(action) == str(1) and str(key) == "generation_4":
    for key in output:
        vidmeshes = output[key]
        for action in range(len(vidmeshes)):
            meshes = vidmeshes[action].transpose(2, 0, 1)
            path = os.path.join(savefolder,
                                "action{}_{}.mp4".format(action, key))
            render_video(meshes, key, action, renderer, path, background)


if __name__ == "__main__":
    main()
