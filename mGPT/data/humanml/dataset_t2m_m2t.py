import random
import numpy as np
from torch.utils import data
from .dataset_t2m import Text2MotionDataset
import codecs as cs
from os.path import join as pjoin


class Text2MotionDatasetM2T(data.Dataset):

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
                
        new_name_list = []
        length_list = []
        data_dict = {}
        for name in self.id_list:
            # try:
                motion = np.load(pjoin(motion_dir, name + '.npy'))
                if (len(motion)) <  self.min_motion_length or (len(motion) >= 200):
                    continue
                
                        
                text_data = []
                flag = False
                
                with cs.open(pjoin(text_dir, name + '.txt')) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split('#')
                        caption = line_split[0]
                        tokens = line_split[1].split(' ')
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict['caption'] = caption
                        text_dict['tokens'] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag*20) : int(to_tag*20)]

                                if (len(n_motion)) < min_motion_length or (len(n_motion) >= 200):
                                    continue
                                
                                new_name = "%s_%f_%f"%(name, f_tag, to_tag)
                                data_dict[new_name] = {'motion': n_motion,
                                                       'length': len(n_motion),
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag, to_tag, name)
                if flag:
                    data_dict[name] = {'motion': motion,
                                    'length': len(motion),
                                    'name': name,
                                    'text': text_data}
                    
                    new_name_list.append(name)
                    length_list.append(len(motion))
            # except:
            #     # Some motion may not exist in KIT dataset
            #     pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list
        self.nfeats = motion.shape[-1]
    
    
    def __len__(self):
        return len(self.data_dict)  
        
    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return name, motion, m_length, True, True, True, True, True, True
