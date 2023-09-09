import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin


class MotionDataset(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        
        # restrian the length of motion and text
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.unit_length = unit_length
        
        # Data mean and std
        self.mean = mean
        self.std = std
        
        # Data path
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, 'new_joint_vecs')
        text_dir = pjoin(data_root, 'texts')

        # Data id list
        self.id_list = []     
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())
                
        # Debug mode
        if tiny or debug:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading HumanML3D {split}",
                ))
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(self.id_list)
            maxdata = 1e10
            subset = ''

        new_name_list = []
        motion_dict = {}

        # Fast loading
        if os.path.exists(pjoin(data_root, f'tmp/{split}{subset}_motion.pkl')):
            with rich.progress.open(pjoin(data_root, f'tmp/{split}{subset}_motion.pkl'),
                    'rb', description=f"Loading HumanML3D {split}") as file:
                motion_dict = pickle.load(file)
            with open(pjoin(data_root, f'tmp/{split}{subset}_index.pkl'), 'rb') as file:
                new_name_list = pickle.load(file)
        else:
            for idx, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                try:
                    motion = [np.load(pjoin(motion_dir, name + ".npy"))]

                    # Read text
                    with cs.open(pjoin(text_dir, name + '.txt')) as f:
                        text_data = []
                        flag = False
                        lines = f.readlines()

                        for line in lines:
                            try:
                                line_split = line.strip().split('#')
                                f_tag = float(line_split[2])
                                to_tag = float(line_split[3])
                                f_tag = 0.0 if np.isnan(f_tag) else f_tag
                                to_tag = 0.0 if np.isnan(to_tag) else to_tag

                                if f_tag == 0.0 and to_tag == 0.0:
                                    flag = True
                                else:
                                    motion_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in motion if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]

                                    if len(motion_new) == 0:
                                        continue
                                    new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                    motion_dict[new_name] = {
                                        'motion': motion_new, 
                                        "length": [len(m[0]) for m in motion_new]}
                                    new_name_list.append(new_name)
                            except:
                                pass

                    if flag:
                        motion_dict[name] = {
                            'motion': motion, 
                            "length": [len(m[0]) for m in motion]}
                        new_name_list.append(name)
                except:
                    pass
                
            if tmpFile:
                os.makedirs(pjoin(data_root, 'tmp'), exist_ok=True)
                
                with open(pjoin(data_root, f'tmp/{split}{subset}_motion.pkl'),'wb') as file:
                    pickle.dump(motion_dict, file)
                with open(pjoin(data_root, f'tmp/{split}{subset}_index.pkl'), 'wb') as file:
                    pickle.dump(new_name_list, file)

        self.motion_dict = motion_dict
        self.name_list = new_name_list
        self.nfeats = motion_dict[new_name_list[0]]['motion'][0].shape[1]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):
        data = self.motion_dict[self.name_list[item]]
        motion_list, m_length = data["motion"], data["length"]

        # Randomly select a motion
        motion = random.choice(motion_list)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        
        # Z Normalization
        motion = (motion - self.mean) / self.std

        return None, motion, m_length, None, None, None, None,
