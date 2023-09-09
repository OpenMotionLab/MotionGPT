import random
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
from .dataset_m import MotionDataset
from .dataset_t2m import Text2MotionDataset


class MotionDatasetVQ(Text2MotionDataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length,
        min_motion_length,
        win_size,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(data_root, split, mean, std, max_motion_length,
                         min_motion_length, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        # Filter out the motions that are too short
        self.window_size = win_size
        name_list = list(self.name_list)
        for name in self.name_list:
            motion = self.data_dict[name]["motion"]
            if motion.shape[0] < self.window_size:
                name_list.remove(name)
                self.data_dict.pop(name)
        self.name_list = name_list

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, length = data["motion"], data["length"]

        idx = random.randint(0, motion.shape[0] - self.window_size)
        motion = motion[idx:idx + self.window_size]
        motion = (motion - self.mean) / self.std

        return None, motion, length, None, None, None, None,
