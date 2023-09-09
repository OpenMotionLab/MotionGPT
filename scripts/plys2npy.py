import os
import time
from argparse import ArgumentParser
from pathlib import Path

import natsort
import numpy as np
import torch
import trimesh
from tqdm import tqdm


def main():
    parser = ArgumentParser()

    group = parser.add_argument_group("Params")
    group.add_argument(
        "--ply_dir",
        type=str,
        required=True,
        help="ply set",
    )
    group.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="output folder",
    )    
    params = parser.parse_args()
    plys2npy(params.ply_dir, params.out_dir)

def plys2npy(ply_dir, out_dir):
    ply_dir = Path(ply_dir)
    paths = []
    file_list = natsort.natsorted(os.listdir(ply_dir))
    for item in file_list:
        if item.endswith(".ply") and not item.endswith("_gt.ply"):
            paths.append(os.path.join(ply_dir, item))


    meshs = np.zeros((len(paths), 6890, 3))
    for i, path in enumerate(paths):
        mesh = trimesh.load_mesh(path, process=False)
        vs = mesh.vertices
        assert vs.shape == (6890, 3)
        meshs[i] = vs 

    basename = os.path.basename(ply_dir)
    if basename.startswith("SMPLFit_"):
        basename = basename[len("SMPLFit_"):]
    file_name = os.path.join(out_dir, basename+ "_mesh.npy")
    np.save(file_name, meshs)
    

if __name__ == "__main__":
    main()
