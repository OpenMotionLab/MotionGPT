from omegaconf import OmegaConf
from os.path import join as pjoin
from mGPT.config import instantiate_from_config


def build_data(cfg, phase="train"):
    data_config = OmegaConf.to_container(cfg.DATASET, resolve=True)
    data_config['params'] = {'cfg': cfg, 'phase': phase}
    if isinstance(data_config['target'], str):
        return instantiate_from_config(data_config)
    elif isinstance(data_config['target'], list):
        data_config_tmp = data_config.copy()
        data_config_tmp['params']['dataModules'] = data_config['target']
        data_config_tmp['target'] = 'mGPT.data.Concat.ConcatDataModule'
        return instantiate_from_config(data_config)
