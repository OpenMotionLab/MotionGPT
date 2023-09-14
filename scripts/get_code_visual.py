import os
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm
from mGPT.config import parse_args
from mGPT.data.build_data import get_datasets
from mGPT.models.build_model import build_model


def main():

    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1
    
    model_name = cfg.model.target.split('.')[-2].lower()
    output_dir = Path(os.path.join(cfg.FOLDER, model_name, cfg.NAME, "tokens_visual_" + cfg.TIME))

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # create dataset
    datasets = get_datasets(cfg, phase="test")[0]
    print("datasets module {} initialized".format("".join(cfg.TRAIN.DATASETS)))

    os.makedirs(output_dir, exist_ok=True)

    # create model
    model = build_model(cfg, datasets)
    if hasattr(model, "motion_vae"):
        model.vae = model.motion_vae
    print("model {} loaded".format(cfg.model.model_type))

    # Strict load vae model
    if cfg.TRAIN.PRETRAINED_VAE:
        state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,
                                map_location="cpu")['state_dict']
        print(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")

        from collections import OrderedDict
        vae_dict = OrderedDict()
        for k, v in state_dict.items():
            if "motion_vae" in k:
                name = k.replace("motion_vae.", "")
                vae_dict[name] = v
            elif "vae" in k:
                name = k.replace("vae.", "")
                vae_dict[name] = v
        if hasattr(model, 'vae'):
            model.vae.load_state_dict(vae_dict, strict=True)
        else:
            model.motion_vae.load_state_dict(vae_dict, strict=True)

    # Strict load pretrianed model
    if cfg.TRAIN.PRETRAINED:
        state_dict = torch.load(cfg.TRAIN.PRETRAINED,
                                map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict, strict=True)

    if cfg.ACCELERATOR == "gpu":
        model = model.cuda()
        
    model.eval()
    codes = cfg.model.codebook_size
    with torch.no_grad():
        for i in tqdm(range(codes)):
            
            # Generate motion from token
            m_token = torch.LongTensor(1, 1).fill_(i).to(model.device)
            # vq_latent = model.vae.quantizer.dequantize(m_token)
            gen_motion = model.vae.decode(m_token)
            gen_motion = model.feats2joints(gen_motion).to('cpu').numpy()
            
            # Generate translation from token
            texts = [f'Generate text: <motion_id_{codes}><motion_id_{i}><motion_id_{codes +1}>']
            # texts = [f'Use only one word to describe: <motion_id_{codes}><motion_id_{i}><motion_id_{codes +1}>']
            batch = {
                "text": texts,
                "length": [0]
            }
            
            out_text = model(batch)['texts']
            print(out_text)
            out_text_path = os.path.join(output_dir, f'{i}.txt')
            Path(out_text_path).parent.mkdir(parents=True, exist_ok=True)
            with open(out_text_path, 'w') as f:
                f.write(out_text[0])
            
            target_path = os.path.join(output_dir, f'{i}.npy')
            Path(target_path).parent.mkdir(parents=True, exist_ok=True)
            
            np.save(target_path, gen_motion)

    print(
        f'Motion tokenization done, the motion tokens are saved to {output_dir}'
    )


if __name__ == "__main__":
    main()
