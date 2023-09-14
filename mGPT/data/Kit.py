import numpy as np
import torch
import os 
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from .humanml.scripts.motion_process import (process_file, recover_from_ric)
from .HumanML3D import HumanML3DDataModule
from .humanml import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, MotionDatasetVQ, Text2MotionDatasetToken


class KitDataModule(HumanML3DDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(cfg, **kwargs)

        # Basic info of the dataset
        self.name = "kit"
        self.njoints = 21

        # Path to the dataset
        data_root = cfg.DATASET.KIT.ROOT
        self.hparams.data_root = data_root
        self.hparams.text_dir = pjoin(data_root, "texts")
        self.hparams.motion_dir = pjoin(data_root, 'new_joint_vecs')

        # Mean and std of the dataset
        dis_data_root = pjoin(cfg.DATASET.KIT.MEAN_STD_PATH, 'kit',
                              "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
        self.hparams.mean = np.load(pjoin(dis_data_root, "mean.npy"))
        self.hparams.std = np.load(pjoin(dis_data_root, "std.npy"))

        # Mean and std for fair evaluation
        dis_data_root_eval = pjoin(cfg.DATASET.KIT.MEAN_STD_PATH, 't2m',
                                   "Comp_v6_KLD005", "meta")
        self.hparams.mean_eval = np.load(pjoin(dis_data_root_eval, "mean.npy"))
        self.hparams.std_eval = np.load(pjoin(dis_data_root_eval, "std.npy"))

        # Length of the dataset
        self.hparams.max_motion_length = cfg.DATASET.KIT.MAX_MOTION_LEN
        self.hparams.min_motion_length = cfg.DATASET.KIT.MIN_MOTION_LEN
        self.hparams.max_text_len = cfg.DATASET.KIT.MAX_TEXT_LEN
        self.hparams.unit_length = cfg.DATASET.KIT.UNIT_LEN
        
        # Get additional info of the dataset
        self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})
        self.nfeats = self._sample_set.nfeats
        cfg.DATASET.NFEATS = self.nfeats

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '03950_gt.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 'kit')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
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
        # random select samples for mm
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
