import torch

def load_pretrained(cfg, model, logger, phase="train"):
    logger.info(f"Loading pretrain model from {cfg.TRAIN.PRETRAINED}")
    if phase == "train":
        ckpt_path = cfg.TRAIN.PRETRAINED
    elif phase == "test":
        ckpt_path = cfg.TEST.CHECKPOINTS
        
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model


def load_pretrained_vae(cfg, model, logger):
    state_dict = torch.load(cfg.TRAIN.PRETRAINED_VAE,
                            map_location="cpu")['state_dict']
    logger.info(f"Loading pretrain vae from {cfg.TRAIN.PRETRAINED_VAE}")
    # Extract encoder/decoder
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
    
    return model
