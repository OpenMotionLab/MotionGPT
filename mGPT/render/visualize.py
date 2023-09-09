from operator import mod
import os
# from cv2 import CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE
import imageio
import shutil
import numpy as np
import torch
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R
from mGPT.render.renderer import get_renderer
from mGPT.render.rendermotion import render_video
# from mld.utils.img_utils import convert_img
# from mld.utils.uicap_utils import output_pkl


def parsename(path):
    basebane = os.path.basename(path)
    base = os.path.splitext(basebane)[0]
    strs = base.split('_')
    key = strs[-2]
    action = strs[-1]
    return key, action


def load_anim(path, timesize=None):
    data = np.array(imageio.mimread(path, memtest=False))  #[..., :3]
    if timesize is None:
        return data

    # take the last frame and put shadow repeat the last frame but with a little shadow
    # lastframe = add_shadow(data[-1])
    # alldata = np.tile(lastframe, (timesize, 1, 1, 1))
    alldata = data

    # debug fix mat dim
    if len(data.shape) == 3 and len(alldata.shape) == 4:
        data = data[:, None, :, :]

    # copy the first frames
    lenanim = data.shape[0]
    alldata[:lenanim] = data[:lenanim]
    return alldata


def plot_3d_motion_dico(x):
    motion, length, save_path, params, kargs = x
    plot_3d_motion(motion, length, save_path, params, **kargs)


def plot_3d_motion(motion,
                   length,
                   save_path,
                   params,
                   title="",
                   interval=50,
                   pred_cam=None,
                   imgs=None,
                   bbox=None,
                   side=None):
    # render smpl
    # [nframes, nVs, 3]
    if motion.shape[1] == 6890:
        # width = 250
        # height = 250
        width = 600
        height = 600
        if pred_cam is None:
            # cam=(0.75, 0.75, 0, 0.1)
            cam = (0.8, 0.8, 0, 0.1)
            # cam=(0.9, 0.9, 0, 0.1)
        else:
            assert bbox is not None
            assert imgs is not None

            # Tmp visulize
            # weak perspective camera parameters in cropped image space (s,tx,ty)
            # to
            # weak perspective camera parameters in original image space (sx,sy,tx,ty)
            cam = np.concatenate(
                (pred_cam[:, [0]], pred_cam[:, [0]], pred_cam[:, 1:3]), axis=1)

            # ToDo convert to original cam
            # load original img?
            # calculate cam after padding???
            #
            # cam = convert_crop_cam_to_orig_img(
            #     cam=pred_cam,
            #     bbox=bbox,
            #     img_width=width,
            #     img_height=height
            # )
        cam_pose = np.eye(4)
        cam_pose[0:3, 0:3] = R.from_euler('x', -90, degrees=True).as_matrix()
        cam_pose[0:3, 3] = [0, 0, 0]
        if side:
            rz = np.eye(4)
            rz[0:3, 0:3] = R.from_euler('z', -90, degrees=True).as_matrix()
            cam_pose = np.matmul(rz, cam_pose)

        # # reshape input imgs
        # if imgs is not None:
        #     imgs = convert_img(imgs.unsqueeze(0), height)[:,0]
        backgrounds = imgs if imgs is not None else np.ones(
            (height, width, 3)) * 255
        renderer = get_renderer(width, height, cam_pose)

        # [nframes, nVs, 3]
        meshes = motion
        key, action = parsename(save_path)
        render_video(meshes,
                     key,
                     action,
                     renderer,
                     save_path,
                     backgrounds,
                     cam_pose,
                     cams=cam)
        return


def stack_images(real, real_gens, gen, real_imgs=None):
    # change to 3 channel
    # print(real.shape)
    # print(real_gens.shape)
    # print(real_gens.shape)
    # real = real[:3]
    # real_gens = real_gens[:3]
    # gen = gen[:3]

    nleft_cols = len(real_gens) + 1
    print("Stacking frames..")
    allframes = np.concatenate(
        (real[:, None, ...], *[x[:, None, ...] for x in real_gens], gen), 1)
    nframes, nspa, nats, h, w, pix = allframes.shape

    blackborder = np.zeros((w // 30, h * nats, pix), dtype=allframes.dtype)
    # blackborder = np.ones((w//30, h*nats, pix), dtype=allframes.dtype)*255
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4,
                                                           0)).transpose(
                                                               3, 1, 0, 2)
        frame = np.concatenate(
            (*columns[0:nleft_cols], blackborder, *columns[nleft_cols:]),
            0).transpose(1, 0, 2)

        frames.append(frame)

    if real_imgs is not None:
        resize_imgs = convert_img(real_imgs, h)[:nframes, ...]

        for i in range(len(frames)):
            imgs = np.vstack(resize_imgs[i, ...])
            imgs4 = np.ones(
                (imgs.shape[0], imgs.shape[1], 4), dtype=np.uint8) * 255
            imgs4[:, :, :3] = imgs
            #imgs = torch2numpy(imgs)
            frames[i] = np.concatenate((imgs4, frames[i]), 1)
    return np.stack(frames)


def stack_images_gen(gen, real_imgs=None):
    print("Stacking frames..")
    allframes = gen
    nframes, nspa, nats, h, w, pix = allframes.shape
    blackborder = np.zeros((w * nspa, h // 30, pix), dtype=allframes.dtype)
    blackborder = blackborder[None, ...].repeat(nats,
                                                axis=0).transpose(0, 2, 1, 3)

    frames = []
    for frame_idx in tqdm(range(nframes)):
        rows = np.vstack(allframes[frame_idx].transpose(0, 3, 2, 4,
                                                        1)).transpose(
                                                            3, 1, 0, 2)
        rows = np.concatenate((rows, blackborder), 1)
        frame = np.concatenate(rows, 0)
        frames.append(frame)

    if real_imgs is not None:
        # ToDo Add images
        resize_imgs = convert_img(real_imgs, h)[:nframes, ...]
        for i in range(len(frames)):
            imgs = np.vstack(resize_imgs[i, ...])
            #imgs = torch2numpy(imgs)
            frames[i] = np.concatenate((imgs, frames[i]), 1)
    return np.stack(frames)


def generate_by_video(visualization, reconstructions, generation,
                      label_to_action_name, params, nats, nspa, tmp_path):
    # shape : (17, 3, 4, 480, 640, 3)
    # (nframes, row, column, h, w, 3)
    fps = params["fps"]

    params = params.copy()

    gen_only = False
    if visualization is None:
        gen_only = True
        outputkey = "output_vertices"
        params["pose_rep"] = "vertices"
    elif "output_vertices" in visualization:
        outputkey = "output_vertices"
        params["pose_rep"] = "vertices"
    elif "output_xyz" in visualization:
        outputkey = "output_xyz"
        params["pose_rep"] = "xyz"
    else:
        outputkey = "poses"

    keep = [outputkey, 'lengths', "y"]
    gener = {key: generation[key].data.cpu().numpy() for key in keep}
    if not gen_only:
        visu = {key: visualization[key].data.cpu().numpy() for key in keep}
        recons = {}
        # visualize regressor results
        if 'vertices_hat' in reconstructions['ntf']:
            recons['regressor'] = {
                'output_vertices':
                reconstructions['ntf']['vertices_hat'].data.cpu().numpy(),
                'lengths':
                reconstructions['ntf']['lengths'].data.cpu().numpy(),
                'y':
                reconstructions['ntf']['y'].data.cpu().numpy()
            }

            recons['regressor_side'] = {
                'output_vertices':
                reconstructions['ntf']['vertices_hat'].data.cpu().numpy(),
                'lengths':
                reconstructions['ntf']['lengths'].data.cpu().numpy(),
                'y':
                reconstructions['ntf']['y'].data.cpu().numpy(),
                'side':
                True
            }
            # ToDo rendering overlap results
            # recons['overlap'] = {'output_vertices':reconstructions['ntf']['vertices_hat'].data.cpu().numpy(),
            #                        'lengths':reconstructions['ntf']['lengths'].data.cpu().numpy(),
            #                        'y':reconstructions['ntf']['y'].data.cpu().numpy(),
            #                        'imgs':reconstructions['ntf']['imgs'],
            #                        'bbox':reconstructions['ntf']['bbox'].data.cpu().numpy(),
            #                        'cam':reconstructions['ntf']['preds'][0]['cam'].data.cpu().numpy()}
        for mode, reconstruction in reconstructions.items():
            recons[mode] = {
                key: reconstruction[key].data.cpu().numpy()
                for key in keep
            }
            recons[mode + '_side'] = {
                key: reconstruction[key].data.cpu().numpy()
                for key in keep
            }
            recons[mode + '_side']['side'] = True

    # lenmax = max(gener['lengths'].max(), visu['lengths'].max())
    # timesize = lenmax + 5 longer visulization
    lenmax = gener['lengths'].max()
    timesize = lenmax

    import multiprocessing

    def pool_job_with_desc(pool, iterator, desc, max_, save_path_format, isij):
        with tqdm(total=max_, desc=desc.format("Render")) as pbar:
            for data in iterator:
                plot_3d_motion_dico(data)
            # for _ in pool.imap_unordered(plot_3d_motion_dico, iterator):
            #     pbar.update()
        if isij:
            array = np.stack([[
                load_anim(save_path_format.format(i, j), timesize)
                for j in range(nats)
            ] for i in tqdm(range(nspa), desc=desc.format("Load"))])
            return array.transpose(2, 0, 1, 3, 4, 5)
        else:
            array = np.stack([
                load_anim(save_path_format.format(i), timesize)
                for i in tqdm(range(nats), desc=desc.format("Load"))
            ])
            return array.transpose(1, 0, 2, 3, 4)

    pool = None
    # if True:
    with multiprocessing.Pool() as pool:
        # Generated samples
        save_path_format = os.path.join(tmp_path, "gen_{}_{}.gif")
        iterator = ((gener[outputkey][i, j], gener['lengths'][i, j],
                     save_path_format.format(i, j), params, {
                         "title":
                         f"gen: {label_to_action_name(gener['y'][i, j])}",
                         "interval": 1000 / fps
                     }) for j in range(nats) for i in range(nspa))
        gener["frames"] = pool_job_with_desc(pool, iterator,
                                             "{} the generated samples",
                                             nats * nspa, save_path_format,
                                             True)
        if not gen_only:
            # Real samples
            save_path_format = os.path.join(tmp_path, "real_{}.gif")
            iterator = ((visu[outputkey][i], visu['lengths'][i],
                         save_path_format.format(i), params, {
                             "title":
                             f"real: {label_to_action_name(visu['y'][i])}",
                             "interval": 1000 / fps
                         }) for i in range(nats))
            visu["frames"] = pool_job_with_desc(pool, iterator,
                                                "{} the real samples", nats,
                                                save_path_format, False)
            for mode, recon in recons.items():
                # Reconstructed samples
                save_path_format = os.path.join(
                    tmp_path, f"reconstructed_{mode}_" + "{}.gif")
                if mode == 'overlap':
                    iterator = ((
                        recon[outputkey][i], recon['lengths'][i],
                        save_path_format.format(i), params, {
                            "title":
                            f"recons: {label_to_action_name(recon['y'][i])}",
                            "interval": 1000 / fps,
                            "pred_cam": recon['cam'][i],
                            "imgs": recon['imgs'][i],
                            "bbox": recon['bbox'][i]
                        }) for i in range(nats))
                else:
                    side = True if 'side' in recon.keys() else False
                    iterator = ((
                        recon[outputkey][i], recon['lengths'][i],
                        save_path_format.format(i), params, {
                            "title":
                            f"recons: {label_to_action_name(recon['y'][i])}",
                            "interval": 1000 / fps,
                            "side": side
                        }) for i in range(nats))
                recon["frames"] = pool_job_with_desc(
                    pool, iterator, "{} the reconstructed samples", nats,
                    save_path_format, False)
    # vis img in visu
    if not gen_only:
        input_imgs = visualization["imgs"] if visualization[
            "imgs"] is not None else None
        vis = visu["frames"] if not gen_only else None
        rec = [recon["frames"]
               for recon in recons.values()] if not gen_only else None
        gen = gener["frames"]
        frames = stack_images(vis, rec, gen, input_imgs)
    else:
        gen = gener["frames"]
        frames = stack_images_gen(gen)
    return frames


def viz_epoch(model,
              dataset,
              epoch,
              params,
              folder,
              module=None,
              writer=None,
              exps=''):
    """ Generate & viz samples """
    module = model if module is None else module

    # visualize with joints3D
    model.outputxyz = True

    print(f"Visualization of the epoch {epoch}")

    noise_same_action = params["noise_same_action"]
    noise_diff_action = params["noise_diff_action"]
    duration_mode = params["duration_mode"]
    reconstruction_mode = params["reconstruction_mode"]
    decoder_test = params["decoder_test"]

    fact = params["fact_latent"]
    figname = params["figname"].format(epoch)

    nspa = params["num_samples_per_action"]
    nats = params["num_actions_to_sample"]

    num_classes = params["num_classes"]
    # nats = min(num_classes, nats)

    # define some classes
    classes = torch.randperm(num_classes)[:nats]
    # duplicate same classes when sampling too much
    if nats > num_classes:
        classes = classes.expand(nats)

    meandurations = torch.from_numpy(
        np.array([
            round(dataset.get_mean_length_label(cl.item())) for cl in classes
        ]))

    if duration_mode == "interpolate" or decoder_test == "diffduration":
        points, step = np.linspace(-nspa, nspa, nspa, retstep=True)
        # points = np.round(10*points/step).astype(int)
        points = np.array([5, 10, 16, 30, 60, 80]).astype(int)
        # gendurations = meandurations.repeat((nspa, 1)) + points[:, None]
        gendurations = torch.from_numpy(points[:, None]).expand(
            (nspa, 1)).repeat((1, nats))
    else:
        gendurations = meandurations.repeat((nspa, 1))
    print("Duration time: ")
    print(gendurations[:, 0])

    # extract the real samples
    # real_samples, real_theta, mask_real, real_lengths, imgs, paths
    batch = dataset.get_label_sample_batch(classes.numpy())

    # ToDo
    # clean these data
    # Visualizaion of real samples
    visualization = {
        "x": batch['x'].to(model.device),
        "y": classes.to(model.device),
        "mask": batch['mask'].to(model.device),
        'lengths': batch['lengths'].to(model.device),
        "output": batch['x'].to(model.device),
        "theta":
        batch['theta'].to(model.device) if 'theta' in batch.keys() else None,
        "imgs":
        batch['imgs'].to(model.device) if 'imgs' in batch.keys() else None,
        "paths": batch['paths'] if 'paths' in batch.keys() else None,
    }

    # Visualizaion of real samples
    if reconstruction_mode == "both":
        reconstructions = {
            "tf": {
                "x":
                batch['x'].to(model.device),
                "y":
                classes.to(model.device),
                'lengths':
                batch['lengths'].to(model.device),
                "mask":
                batch['mask'].to(model.device),
                "teacher_force":
                True,
                "theta":
                batch['theta'].to(model.device)
                if 'theta' in batch.keys() else None
            },
            "ntf": {
                "x":
                batch['x'].to(model.device),
                "y":
                classes.to(model.device),
                'lengths':
                batch['lengths'].to(model.device),
                "mask":
                batch['mask'].to(model.device),
                "theta":
                batch['theta'].to(model.device)
                if 'theta' in batch.keys() else None
            }
        }
    else:
        reconstructions = {
            reconstruction_mode: {
                "x":
                batch['x'].to(model.device),
                "y":
                classes.to(model.device),
                'lengths':
                batch['lengths'].to(model.device),
                "mask":
                batch['mask'].to(model.device),
                "teacher_force":
                reconstruction_mode == "tf",
                "imgs":
                batch['imgs'].to(model.device)
                if 'imgs' in batch.keys() else None,
                "theta":
                batch['theta'].to(model.device)
                if 'theta' in batch.keys() else None,
                "bbox":
                batch['bbox'] if 'bbox' in batch.keys() else None
            }
        }
    print("Computing the samples poses..")

    # generate the repr (joints3D/pose etc)
    model.eval()
    with torch.no_grad():
        # Reconstruction of the real data
        for mode in reconstructions:
            # update reconstruction dicts
            reconstructions[mode] = model(reconstructions[mode])
        reconstruction = reconstructions[list(reconstructions.keys())[0]]

        if decoder_test == "gt":
            # Generate the new data
            gt_input = {
                "x": batch['x'].repeat(nspa, 1, 1, 1).to(model.device),
                "y": classes.repeat(nspa).to(model.device),
                "mask": batch['mask'].repeat(nspa, 1).to(model.device),
                'lengths': batch['lengths'].repeat(nspa).to(model.device)
            }
            generation = model(gt_input)
        if decoder_test == "new":
            # Generate the new data
            generation = module.generate(gendurations,
                                         classes=classes,
                                         nspa=nspa,
                                         noise_same_action=noise_same_action,
                                         noise_diff_action=noise_diff_action,
                                         fact=fact)
        elif decoder_test == "diffaction":
            assert nats == nspa
            # keep the same noise for each "sample"
            z = reconstruction["z"].repeat((nspa, 1))
            mask = reconstruction["mask"].repeat((nspa, 1))
            lengths = reconstruction['lengths'].repeat(nspa)
            # but use other labels
            y = classes.repeat_interleave(nspa).to(model.device)
            generation = {"z": z, "y": y, "mask": mask, 'lengths': lengths}
            model.decoder(generation)

        elif decoder_test == "diffduration":
            z = reconstruction["z"].repeat((nspa, 1))
            lengths = gendurations.reshape(-1).to(model.device)
            mask = model.lengths_to_mask(lengths)
            y = classes.repeat(nspa).to(model.device)
            generation = {"z": z, "y": y, "mask": mask, 'lengths': lengths}
            model.decoder(generation)

        elif decoder_test == "interpolate_action":
            assert nats == nspa
            # same noise for each sample
            z_diff_action = torch.randn(1,
                                        model.latent_dim,
                                        device=model.device).repeat(nats, 1)
            z = z_diff_action.repeat((nspa, 1))

            # but use combination of labels and labels below
            y = F.one_hot(classes.to(model.device),
                          model.num_classes).to(model.device)
            y_below = F.one_hot(torch.cat((classes[1:], classes[0:1])),
                                model.num_classes).to(model.device)
            convex_factors = torch.linspace(0, 1, nspa, device=model.device)
            y_mixed = torch.einsum("nk,m->mnk", y, 1-convex_factors) + \
                torch.einsum("nk,m->mnk", y_below, convex_factors)
            y_mixed = y_mixed.reshape(nspa * nats, y_mixed.shape[-1])

            durations = gendurations[0].to(model.device)
            durations_below = torch.cat((durations[1:], durations[0:1]))

            gendurations = torch.einsum("l,k->kl", durations, 1-convex_factors) + \
                torch.einsum("l,k->kl", durations_below, convex_factors)
            gendurations = gendurations.to(dtype=durations.dtype)

            lengths = gendurations.to(model.device).reshape(z.shape[0])
            mask = model.lengths_to_mask(lengths)

            generation = {
                "z": z,
                "y": y_mixed,
                "mask": mask,
                'lengths': lengths
            }
            generation = model.decoder(generation)

        visualization = module.prepare(visualization)
        visualization["output_xyz"] = visualization["x_xyz"]
        visualization["output_vertices"] = visualization["x_vertices"]
        # Get xyz for the real ones
        # visualization["output_xyz"] = module.rot2xyz(visualization["output"], visualization["mask"], jointstype="smpl")
        # # Get smpl vertices for the real ones
        # if module.cvae.pose_rep != "xyz":
        #     visualization["output_vertices"] = module.rot2xyz(visualization["output"], visualization["mask"], jointstype="vertices")

    for key, val in generation.items():
        if len(generation[key].shape) == 1:
            generation[key] = val.reshape(nspa, nats)
        else:
            generation[key] = val.reshape(nspa, nats, *val.shape[1:])

    finalpath = os.path.join(folder, figname + exps + ".gif")
    tmp_path = os.path.join(folder, f"subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    print("Generate the videos..")
    frames = generate_by_video(visualization, reconstructions, generation,
                               dataset.label_to_action_name, params, nats,
                               nspa, tmp_path)

    print(f"Writing video {finalpath}")
    imageio.mimsave(finalpath.replace('gif', 'mp4'), frames, fps=params["fps"])
    shutil.rmtree(tmp_path)

    # output npy
    output = {
        "data_id": batch['id'],
        "paths": batch['paths'],
        "x": batch['x'].cpu().numpy(),
        "x_vertices": visualization["x_vertices"].cpu().numpy(),
        "output_vertices":
        reconstructions['ntf']["output_vertices"].cpu().numpy(),
        "gen_vertices": generation["output_vertices"].cpu().numpy()
    }

    outputpath = finalpath.replace('gif', 'npy')
    np.save(outputpath, output)

    # output pkl
    batch_recon = reconstructions["ntf"]
    outputpath = finalpath.replace('gif', 'pkl')
    # output_pkl([batch_recon], outputpath)

    if writer is not None:
        writer.add_video(f"Video/Epoch {epoch}",
                         frames.transpose(0, 3, 1, 2)[None],
                         epoch,
                         fps=params["fps"])
    return finalpath


def viz_dataset(dataset, params, folder):
    """ Generate & viz samples """
    print("Visualization of the dataset")

    nspa = params["num_samples_per_action"]
    nats = params["num_actions_to_sample"]

    num_classes = params["num_classes"]

    figname = "{}_{}_numframes_{}_sampling_{}_step_{}".format(
        params["dataset"], params["pose_rep"], params["num_frames"],
        params["sampling"], params["sampling_step"])

    # define some classes
    classes = torch.randperm(num_classes)[:nats]

    allclasses = classes.repeat(nspa, 1).reshape(nspa * nats)
    # extract the real samples
    real_samples, mask_real, real_lengths = dataset.get_label_sample_batch(
        allclasses.numpy())
    # to visualize directly

    # Visualizaion of real samples
    visualization = {
        "x": real_samples,
        "y": allclasses,
        "mask": mask_real,
        'lengths': real_lengths,
        "output": real_samples
    }

    from mGPT.models.rotation2xyz import Rotation2xyz

    device = params["device"]
    rot2xyz = Rotation2xyz(device=device)

    rot2xyz_params = {
        "pose_rep": params["pose_rep"],
        "glob_rot": params["glob_rot"],
        "glob": params["glob"],
        "jointstype": params["jointstype"],
        "translation": params["translation"]
    }

    output = visualization["output"]
    visualization["output_xyz"] = rot2xyz(output.to(device),
                                          visualization["mask"].to(device),
                                          **rot2xyz_params)

    for key, val in visualization.items():
        if len(visualization[key].shape) == 1:
            visualization[key] = val.reshape(nspa, nats)
        else:
            visualization[key] = val.reshape(nspa, nats, *val.shape[1:])

    finalpath = os.path.join(folder, figname + ".gif")
    tmp_path = os.path.join(folder, f"subfigures_{figname}")
    os.makedirs(tmp_path, exist_ok=True)

    print("Generate the videos..")
    frames = generate_by_video_sequences(visualization,
                                         dataset.label_to_action_name, params,
                                         nats, nspa, tmp_path)

    print(f"Writing video {finalpath}..")
    imageio.mimsave(finalpath, frames, fps=params["fps"])


def generate_by_video_sequences(visualization, label_to_action_name, params,
                                nats, nspa, tmp_path):
    # shape : (17, 3, 4, 480, 640, 3)
    # (nframes, row, column, h, w, 3)
    fps = params["fps"]
    if "output_vetices" in visualization:
        outputkey = "output_vetices"
        params["pose_rep"] = "vertices"
    elif "output_xyz" in visualization:
        outputkey = "output_xyz"
        params["pose_rep"] = "xyz"
    else:
        outputkey = "poses"

    keep = [outputkey, 'lengths', "y"]
    visu = {key: visualization[key].data.cpu().numpy() for key in keep}
    lenmax = visu['lengths'].max()

    timesize = lenmax + 5

    # import multiprocessing

    def pool_job_with_desc(pool, iterator, desc, max_, save_path_format):
        for data in iterator:
            plot_3d_motion_dico(data)
        # with tqdm(total=max_, desc=desc.format("Render")) as pbar:
        #     for _ in pool.imap_unordered(plot_3d_motion_dico, iterator):
        #         pbar.update()
        array = np.stack([[
            load_anim(save_path_format.format(i, j), timesize)
            for j in range(nats)
        ] for i in tqdm(range(nspa), desc=desc.format("Load"))])
        return array.transpose(2, 0, 1, 3, 4, 5)

    pool = None
    # with multiprocessing.Pool() as pool:
    # Real samples
    save_path_format = os.path.join(tmp_path, "real_{}_{}.gif")
    iterator = ((visu[outputkey][i, j], visu['lengths'][i, j],
                 save_path_format.format(i, j), params, {
                     "title": f"real: {label_to_action_name(visu['y'][i, j])}",
                     "interval": 1000 / fps
                 }) for j in range(nats) for i in range(nspa))
    visu["frames"] = pool_job_with_desc(pool, iterator, "{} the real samples",
                                        nats, save_path_format)
    frames = stack_images_sequence(visu["frames"])
    return frames


def stack_images_sequence(visu):
    print("Stacking frames..")
    allframes = visu
    nframes, nspa, nats, h, w, pix = allframes.shape
    frames = []
    for frame_idx in tqdm(range(nframes)):
        columns = np.vstack(allframes[frame_idx].transpose(1, 2, 3, 4,
                                                           0)).transpose(
                                                               3, 1, 0, 2)
        frame = np.concatenate(columns).transpose(1, 0, 2)
        frames.append(frame)
    return np.stack(frames)
