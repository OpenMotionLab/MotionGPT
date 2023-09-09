# inspired and modified by Mathis Petrovich from
# https://github.com/vchoutas/smplx/tree/master/tools

# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import os.path as osp
import pickle

import argparse

import numpy as np


def remove_chumpy_dep(dico):
    output_dict = {}
    for key, val in dico.items():
        if 'chumpy' in str(type(val)):
            output_dict[key] = np.array(val)
        else:
            output_dict[key] = val
    return output_dict


def load_and_remove_chumpy_dep(path):
    with open(path, 'rb') as pkl_file:
        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        data = pickle.load(pkl_file, encoding="latin1")

    data = remove_chumpy_dep(data)
    return data


def load_npz_into_dict(path):
    data = {key: val for key, val in np.load(smplh_fn).items()}
    data = remove_chumpy_dep(data)
    return data


def load_and_clean_data(path):
    ext = os.path.splitext(path)[-1]
    if ext == ".npz":
        data = load_npz_into_dict(path)
    elif ext == ".pkl":
        data = load_and_remove_chumpy_dep(path)
    else:
        raise TypeError("The format should be pkl or npz")
    return data


def merge_models(smplh_fn, mano_left_fn, mano_right_fn,
                 output_folder='output'):

    body_data = load_and_clean_data(smplh_fn)
    lhand_data = load_and_clean_data(mano_left_fn)
    rhand_data = load_and_clean_data(mano_right_fn)

    modelname = osp.split(smplh_fn)[1]
    parent_folder = osp.split(osp.split(smplh_fn)[0])[1]
    if "female" in parent_folder + "_" + modelname.lower():
        out_fn = "SMPLH_FEMALE.npz"
    elif "male" in parent_folder + "_" + modelname.lower():
        out_fn = "SMPLH_MALE.npz"
    elif "neutral" in parent_folder + "_" + modelname.lower():
        out_fn = "SMPLH_NEUTRAL.npz"
    else:
        out_fn = modelname

    output_data = body_data.copy()
    output_data['hands_componentsl'] = lhand_data['hands_components']
    output_data['hands_componentsr'] = rhand_data['hands_components']

    output_data['hands_coeffsl'] = lhand_data['hands_coeffs']
    output_data['hands_coeffsr'] = rhand_data['hands_coeffs']

    output_data['hands_meanl'] = lhand_data['hands_mean']
    output_data['hands_meanr'] = rhand_data['hands_mean']

    # Just in case
    output_data = remove_chumpy_dep(output_data)

    out_path = osp.join(output_folder, out_fn)
    print('Saving to {}'.format(out_path))

    # np.savez(out_path, output_data)
    np.savez_compressed(out_path, **output_data)
    # with open(out_path, 'wb') as output_file:
    #     pickle.dump(output_data, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smplh-fn', dest='smplh_fn', required=True,
                        type=str, help='The path to the SMPLH model')
    parser.add_argument('--mano-left-fn', dest='mano_left_fn', required=True,
                        type=str, help='The path to the left hand MANO model')
    parser.add_argument('--mano-right-fn', dest='mano_right_fn', required=True,
                        type=str, help='The path to the right hand MANO model')
    parser.add_argument('--output-folder', dest='output_folder',
                        required=True, type=str,
                        help='The path to the output folder')

    args = parser.parse_args()

    smplh_fn = args.smplh_fn
    mano_left_fn = args.mano_left_fn
    mano_right_fn = args.mano_right_fn
    output_folder = args.output_folder

    if not osp.exists(output_folder):
        print('Creating directory: {}'.format(output_folder))
        os.makedirs(output_folder)

    merge_models(smplh_fn, mano_left_fn, mano_right_fn, output_folder)
