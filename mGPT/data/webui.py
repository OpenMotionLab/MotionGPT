import numpy as np
import torch
from os.path import join as pjoin
from .humanml.scripts.motion_process import (process_file, recover_from_ric)
from . import BASEDataModule
from .humanml import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, MotionDatasetVQ, Text2MotionDatasetToken, Text2MotionDatasetM2T
from .utils import humanml3d_collate


class HumanML3DDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=humanml3d_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'humanml3d'
        self.name = "humanml3d"
        self.njoints = 22
        
        # Path to the dataset
        data_root = cfg.DATASET.HUMANML3D.ROOT
        self.hparams.data_root = data_root
        self.hparams.text_dir = pjoin(data_root, "texts")
        self.hparams.motion_dir = pjoin(data_root, 'new_joint_vecs')
        
        # Mean and std of the dataset
        self.hparams.mean = np.load(pjoin('assets/meta', "mean.npy"))
        self.hparams.std = np.load(pjoin('assets/meta', "std.npy"))
        
        # Mean and std for fair evaluation
        self.hparams.mean_eval = np.load(pjoin('assets/meta', "mean_eval.npy"))
        self.hparams.std_eval = np.load(pjoin('assets/meta', "std_eval.npy"))
        
        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.HUMANML3D.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.HUMANML3D.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.HUMANML3D.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.HUMANML3D.UNIT_LEN

        # Additional parameters
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE

        # Dataset switch
        self.DatasetEval = Text2MotionDatasetEval

        if cfg.TRAIN.STAGE == "vae":
            if cfg.model.params.motion_vae.target.split('.')[-1].lower() == "vqvae":
                self.hparams.win_size = 64
                self.Dataset = MotionDatasetVQ
            else:
                self.Dataset = MotionDataset
        elif 'lm' in cfg.TRAIN.STAGE:
            self.hparams.code_path = cfg.DATASET.CODE_PATH
            self.hparams.task_path = cfg.DATASET.TASK_PATH
            self.hparams.std_text = cfg.DATASET.HUMANML3D.STD_TEXT
            self.Dataset = Text2MotionDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = Text2MotionDatasetToken
            self.DatasetEval = Text2MotionDatasetToken
        elif cfg.TRAIN.STAGE == "m2t":
            self.Dataset = Text2MotionDatasetM2T
            self.DatasetEval = Text2MotionDatasetM2T
        else:
            self.Dataset = Text2MotionDataset

        # Get additional info of the dataset
        self.nfeats = 263
        cfg.DATASET.NFEATS = self.nfeats
        

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list
