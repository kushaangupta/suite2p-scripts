import numpy as np
import os
from natsort import natsorted
from suite2p.io import save  # Import suite2p as s2p in your environment
import vr2p
import matplotlib.pyplot as plt


animal = 'Tyche-A7'
# load data.
path = f'data/processed/{animal}/multi_day_demix/vr2p.zarr'
place_field_cache_path = f'data/processed/{animal}/multi_day_demix/placefield.zarr'

data = vr2p.ExperimentData(path)

def extract_individual_maxprojs(max_proj, save_folder):
    """
    Extracts individual max projections from a composite max projection image.
    
    Parameters:
    max_proj (np.ndarray): Composite max projection image
    save_folder (str): Path to folder containing plane*/ subdirectories
    
    Returns:
    list: List of extracted max projections for each plane
    """
    # Get sorted plane folders
    plane_folders = natsorted([
        f.path for f in os.scandir(save_folder) 
        if f.is_dir() and f.name.startswith("plane")
    ])

    # Load all ops files
    ops_list = [
        np.load(os.path.join(folder, "ops.npy"), allow_pickle=True).item() 
        for folder in plane_folders
    ]

    # Calculate offsets using suite2p's method
    dy, dx = save.compute_dydx(ops_list)

    individual_maxprojs = []
    for k, ops in enumerate(ops_list):
        # Calculate original position in composite image
        x0 = dx[k] + ops["xrange"][0]
        x1 = dx[k] + ops["xrange"][-1]
        y0 = dy[k] + ops["yrange"][0]
        y1 = dy[k] + ops["yrange"][-1]
        print(x0, x1, y0, y1)

        # Extract the individual max projection
        plane_max = max_proj[y0:y1, x0:x1].copy()

        # Update ops dictionary and save
        ops["max_proj"] = plane_max

        individual_maxprojs.append(plane_max)

    return individual_maxprojs

basepath_animal = 'data/processed/Tyche-A7/'
session_dirs = [
    '2021_12_27/1', '2021_12_28/2', '2021_12_29/1', '2021_12_30/1',
    '2021_12_31/1', '2022_01_03/1', '2022_01_04/1', '2022_01_05/1',
    '2022_01_06/1'
]
results_dir = './results'
os.makedirs(results_dir, exist_ok=True)
for i, session_dir in enumerate(session_dirs):
    save_path = os.path.join(basepath_animal, session_dir, 'suite2p')
    if not os.path.exists(save_path):
        print(f"Session path does not exist: {save_path}")
    original_maxprojs = extract_individual_maxprojs(
        data.images.original[i]['max_img'],
        save_path
    )
    registered_maxprojs = extract_individual_maxprojs(
        data.images.registered[i]['max_img'],
        save_path
    )
    session_path = os.path.join(results_dir, f"session_{i}")
    os.makedirs(session_path)
    # save individual max projections in a session folder as png
    for j, (orig_max, reg_max) in enumerate(zip(original_maxprojs, registered_maxprojs)):
        orig_save_path = os.path.join(session_path, f"plane_{j}_original_max_proj.png")
        reg_save_path = os.path.join(session_path, f"plane_{j}_registered_max_proj.png")
        print(f"Saving original max projection to {orig_save_path}")
        plt.imsave(orig_save_path, orig_max)
        print(f"Saving registered max projection to {reg_save_path}")
        plt.imsave(reg_save_path, reg_max)
