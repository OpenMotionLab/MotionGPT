import json
import os
from pathlib import Path
import time
import numpy as np
import pytorch_lightning as pl
import torch
from rich import get_console
from rich.table import Table
from omegaconf import OmegaConf
from tqdm import tqdm
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
from mGPT.models.build_model import build_model
from mGPT.utils.logger import create_logger
import mGPT.render.matplot.plot_3d_global as plot_3d


def motion_token_to_string(motion_token, lengths, codebook_size=512):
    motion_string = []
    for i in range(motion_token.shape[0]):
        motion_i = motion_token[i].cpu(
        ) if motion_token.device.type == 'cuda' else motion_token[i]
        motion_list = motion_i.tolist()[:lengths[i]]
        motion_string.append(
            (f'<motion_id_{codebook_size}>' +
             ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
             f'<motion_id_{codebook_size + 1}>'))
    return motion_string


def load_example_input(txt_path, task, model):
    with open(txt_path, "r") as file:
        Lines = file.readlines()
    Lines = [line for line in Lines if line.strip()]
    count = 0
    texts = []
    # Strips the newline character
    motion_joints = [torch.zeros((1, 1, 22, 3))] * len(Lines)
    motion_lengths = [0] * len(Lines)
    motion_token_string = ['']
    motion_head = []
    motion_heading = []
    motion_tailing = []
    motion_token = torch.zeros((1, 263))
    for i, line in enumerate(Lines):
        count += 1
        if len(line.split('#')) == 1:
            texts.append(line)
        else:
            feat_path = line.split('#')[1].replace('\n', '')
            if os.path.exists(feat_path):
                feats = torch.tensor(np.load(feat_path), device=model.device)
                feats = model.datamodule.normalize(feats)

                motion_lengths[i] = feats.shape[0]
                motion_token, _ = model.vae.encode(feats[None])

                motion_token_string = motion_token_to_string(
                    motion_token, [motion_token.shape[1]])[0]
                motion_token_length = motion_token.shape[1]

                motion_splited = motion_token_string.split('>')

                split = motion_token_length // 5 + 1
                split2 = motion_token_length // 4 + 1
                split3 = motion_token_length // 4 * 3 + 1

                motion_head.append(motion_token[:, :motion_token.shape[1] //
                                                5][0])

                motion_heading.append(feats[:feats.shape[0] // 4])

                motion_tailing.append(feats[feats.shape[0] // 4 * 3:])

                if '<Motion_Placeholder_s1>' in line:
                    motion_joints[i] = model.feats2joints(
                        feats)[:, :feats.shape[1] // 5]
                else:
                    motion_joints[i] = model.feats2joints(feats)

                motion_split1 = '>'.join(
                    motion_splited[:split]
                ) + f'><motion_id_{model.codebook_size+1}>'
                motion_split2 = f'<motion_id_{model.codebook_size}>' + '>'.join(
                    motion_splited[split:])

                motion_masked = '>'.join(
                    motion_splited[:split2]
                ) + '>' + f'<motion_id_{model.codebook_size+2}>' * (
                    split3 - split2) + '>'.join(motion_splited[split3:])

            texts.append(
                line.split('#')[0].replace(
                    '<motion>', motion_token_string).replace(
                        '<Motion_Placeholder_s1>', motion_split1).replace(
                            '<Motion_Placeholder_s2>', motion_split2).replace(
                                '<Motion_Placeholder_Masked>', motion_masked))

    return_dict = {
        'text': texts,
        'motion_joints': motion_joints,
        'motion_lengths': motion_lengths,
        'motion_token': motion_token,
        'motion_token_string': motion_token_string,
    }
    if len(motion_head) > 0:
        return_dict['motion_head'] = motion_head

    if len(motion_heading) > 0:
        return_dict['motion_heading'] = motion_heading

    if len(motion_tailing) > 0:
        return_dict['motion_tailing'] = motion_tailing

    return return_dict


def main():
    # parse options
    cfg = parse_args(phase="demo")  # parse config file
    cfg.FOLDER = cfg.TEST.FOLDER

    # create logger
    logger = create_logger(cfg, phase="test")

    task = cfg.DEMO.TASK
    text = None

    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.target.split('.')[-2]), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(OmegaConf.to_yaml(cfg))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")

    # Dataset
    datamodule = build_data(cfg)
    logger.info("datasets module {} initialized".format("".join(
        cfg.DATASET.target.split('.')[-2])))

    # create model
    total_time = time.time()
    model = build_model(cfg, datamodule)
    logger.info("model {} loaded".format(cfg.model.target))

    # loading state dict
    if cfg.TEST.CHECKPOINTS:
        logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
        state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                                map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
    else:
        logger.warning(
            "No checkpoints provided, using random initialized model")

    model.to(device)

    if cfg.DEMO.EXAMPLE:
        # Check txt file input
        # load txt
        return_dict = load_example_input(cfg.DEMO.EXAMPLE, task, model)
        text, in_joints = return_dict['text'], return_dict['motion_joints']

    batch_size = 64
    if text:
        for b in tqdm(range(len(text) // batch_size + 1)):
            text_batch = text[b * batch_size:(b + 1) * batch_size]
            in_joints_batch = in_joints[b * batch_size:(b + 1) * batch_size]
            batch = {
                "length":
                return_dict["motion_lengths"][b * batch_size:(b + 1) *
                                              batch_size],
                "text":
                text_batch
            }
            if 'motion_head' in return_dict:
                batch["motion"] = return_dict['motion_head'][b *
                                                             batch_size:(b +
                                                                         1) *
                                                             batch_size]
            if 'motion_heading' in return_dict:
                batch["motion_heading"] = return_dict['motion_heading'][
                    b * batch_size:(b + 1) * batch_size]
            if 'motion_tailing' in return_dict:
                batch["motion_tailing"] = return_dict['motion_tailing'][
                    b * batch_size:(b + 1) * batch_size]

            outputs = model(batch, task=cfg.model.params.task)
            logger.info('Model forward finished! Start saving results...')
            joints = outputs["joints"]
            lengths = outputs["length"]
            output_texts = outputs["texts"]

            for i in range(len(joints)):
                xyz = joints[i][:lengths[i]]
                xyz = xyz[None]

                try:
                    xyz = xyz.detach().cpu().numpy()
                    xyz_in = in_joints_batch[i][None].detach().cpu().numpy()
                except:
                    xyz = xyz.detach().numpy()
                    xyz_in = in_joints[i][None].detach().numpy()

                id = b * batch_size + i

                np.save(os.path.join(output_dir, f'{id}_out.npy'), xyz)
                np.save(os.path.join(output_dir, f'{id}_in.npy'), xyz_in)

                with open(os.path.join(output_dir, f'{id}_in.txt'), 'w') as f:
                    f.write(text_batch[i])

                with open(os.path.join(output_dir, f'{id}_out.txt'), 'w') as f:
                    f.write(output_texts[i])

                # pose_vis = plot_3d.draw_to_batch(xyz_in, [''], [os.path.join(output_dir, f'{i}_in.gif')])
                # pose_vis = plot_3d.draw_to_batch(xyz, [''], [os.path.join(output_dir, f'{i}_out.gif')])

    total_time = time.time() - total_time
    logger.info(
        f'Total time spent: {total_time:.2f} seconds (including model loading time and exporting time).'
    )
    logger.info(f"Testing done, the npy are saved to {output_dir}")


if __name__ == "__main__":
    main()
