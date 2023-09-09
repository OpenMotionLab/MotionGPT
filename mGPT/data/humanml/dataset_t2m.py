import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin


class Text2MotionDataset(data.Dataset):

    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=40,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):

        # restrian the length of motion and text
        self.max_length = 20
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
            enumerator = enumerate(self.id_list)
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading HumanML3D {split}",
                ))
            maxdata = 1e10
            subset = ''

        new_name_list = []
        length_list = []
        data_dict = {}

        # Fast loading
        if os.path.exists(pjoin(data_root, f'tmp/{split}{subset}_data.pkl')):
            if tiny or debug:
                with open(pjoin(data_root, f'tmp/{split}{subset}_data.pkl'),
                          'rb') as file:
                    data_dict = pickle.load(file)
            else:
                with rich.progress.open(
                        pjoin(data_root, f'tmp/{split}{subset}_data.pkl'),
                        'rb',
                        description=f"Loading HumanML3D {split}") as file:
                    data_dict = pickle.load(file)
            with open(pjoin(data_root, f'tmp/{split}{subset}_index.pkl'),
                      'rb') as file:
                name_list = pickle.load(file)
            for name in new_name_list:
                length_list.append(data_dict[name]['length'])

        else:
            for idx, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                try:
                    motion = np.load(pjoin(motion_dir, name + ".npy"))
                    if (len(motion)) < self.min_motion_length or (len(motion)
                                                                  >= 200):
                        continue

                    # Read text
                    text_data = []
                    flag = False
                    with cs.open(pjoin(text_dir, name + '.txt')) as f:
                        lines = f.readlines()
                        for line in lines:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                motion_new = motion[int(f_tag *
                                                        fps):int(to_tag * fps)]
                                if (len(motion_new)
                                    ) < self.min_motion_length or (
                                        len(motion_new) >= 200):
                                    continue
                                new_name = random.choice(
                                    'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                while new_name in new_name_list:
                                    new_name = random.choice(
                                        'ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                name_count = 1
                                while new_name in data_dict:
                                    new_name += '_' + name_count
                                    name_count += 1
                                data_dict[new_name] = {
                                    'motion': motion_new,
                                    "length": len(motion_new),
                                    'text': [text_dict]
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(motion_new))

                    if flag:
                        data_dict[name] = {
                            'motion': motion,
                            "length": len(motion),
                            'text': text_data
                        }
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except:
                    pass

            name_list, length_list = zip(
                *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

            if tmpFile:
                os.makedirs(pjoin(data_root, 'tmp'), exist_ok=True)
                with open(pjoin(data_root, f'tmp/{split}{subset}_data.pkl'),
                          'wb') as file:
                    pickle.dump(data_dict, file)
                with open(pjoin(data_root, f'tmp/{split}{subset}_index.pkl'),
                          'wb') as file:
                    pickle.dump(name_list, file)

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.nfeats = data_dict[name_list[0]]['motion'].shape[1]
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data["caption"]

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

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

        return caption, motion, m_length, None, None, None, None, all_captions
