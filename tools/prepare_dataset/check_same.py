import os
import numpy as np
from tqdm import tqdm



source_root = "/home/jackychou/dataset/UAVRadar"
target_root = "/home/jackychou/Zhou/dataset/UAVRadar"
uav_ids = [i for i in range(1, 1 + 1)]

for uav_id in uav_ids:
    for frame_id in tqdm(range(0, 300)):
        for chirp_id in [i for i in range(1, 255 + 1)]:
            source_path = os.path.join(source_root, f"uav_seqs_{uav_id}", "python_slice_frame_11/azimuth/raw_frame_RA",
                                       f"{frame_id:03d}_{chirp_id:09d}.npy")
            target_path = os.path.join(target_root, f"uav_seqs_{uav_id}", "python_slice_frame_11/azimuth/raw_frame_RA",
                                       f"{frame_id:03d}_{chirp_id:09d}.npy")
            source_file = np.load(source_path)
            target_file = np.load(target_path)
            if np.array_equal(source_file, target_file) == False:
                break
        if np.array_equal(source_file, target_file) == False:
            break
    if np.array_equal(source_file, target_file) == False:
        print("###!")
        break





