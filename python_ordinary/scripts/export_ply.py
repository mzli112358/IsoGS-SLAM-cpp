import os
import re
import argparse
from importlib.machinery import SourceFileLoader

import numpy as np
from plyfile import PlyData, PlyElement

# Spherical harmonic constant
C0 = 0.28209479177387814


def rgb_to_spherical_harmonic(rgb):
    return (rgb-0.5) / C0


def spherical_harmonic_to_rgb(sh):
    return sh*C0 + 0.5


def save_ply(path, means, scales, rotations, rgbs, opacities, normals=None):
    if normals is None:
        normals = np.zeros_like(means)

    colors = rgb_to_spherical_harmonic(rgbs)

    if scales.shape[1] == 1:
        scales = np.tile(scales, (1, 3))

    attrs = ['x', 'y', 'z',
             'nx', 'ny', 'nz',
             'f_dc_0', 'f_dc_1', 'f_dc_2',
             'opacity',
             'scale_0', 'scale_1', 'scale_2',
             'rot_0', 'rot_1', 'rot_2', 'rot_3',]

    dtype_full = [(attribute, 'f4') for attribute in attrs]
    elements = np.empty(means.shape[0], dtype=dtype_full)

    attributes = np.concatenate((means, normals, colors, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

    print(f"Saved PLY format Splat to {path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load SplaTAM config
    experiment = SourceFileLoader(os.path.basename(args.config), args.config).load_module()
    config = experiment.config
    work_path = config['workdir']
    run_name = config['run_name']
    result_dir = os.path.join(work_path, run_name)
    
    # Check if params.npz exists (final result)
    params_npz_path = os.path.join(result_dir, "params.npz")
    
    if os.path.exists(params_npz_path):
        params_path = params_npz_path
        print(f"Found final params file: {params_path}")
    else:
        # Find the latest checkpoint file (params{数字}.npz)
        pattern = re.compile(r'^params(\d+)\.npz$')
        checkpoint_files = []
        
        if os.path.exists(result_dir):
            for filename in os.listdir(result_dir):
                match = pattern.match(filename)
                if match:
                    checkpoint_num = int(match.group(1))
                    checkpoint_files.append((checkpoint_num, filename))
        
        if checkpoint_files:
            # Sort by checkpoint number and get the latest
            checkpoint_files.sort(key=lambda x: x[0], reverse=True)
            latest_checkpoint = checkpoint_files[0]
            params_path = os.path.join(result_dir, latest_checkpoint[1])
            print(f"Found latest checkpoint file: {params_path} (frame {latest_checkpoint[0]})")
        else:
            raise FileNotFoundError(f"No params file found in {result_dir}. Please check if the experiment has been run.")

    params = dict(np.load(params_path, allow_pickle=True))
    means = params['means3D']
    scales = params['log_scales']
    rotations = params['unnorm_rotations']
    rgbs = params['rgb_colors']
    opacities = params['logit_opacities']

    # Generate output PLY filename based on input
    if params_path.endswith("params.npz"):
        ply_filename = "splat.ply"
    else:
        # Extract checkpoint number from filename
        checkpoint_num = os.path.basename(params_path).replace("params", "").replace(".npz", "")
        ply_filename = f"splat_{checkpoint_num}.ply"
    
    ply_path = os.path.join(result_dir, ply_filename)

    save_ply(ply_path, means, scales, rotations, rgbs, opacities)