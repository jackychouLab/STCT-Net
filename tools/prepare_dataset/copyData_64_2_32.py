import os
from tqdm import tqdm
import numpy as np



def check_path(x):
    if type(x) == str:
        x = [x]
    for sub_x in x:
        if os.path.exists(sub_x) is False:
            os.mkdir(sub_x)


source_root = "/home/jackychou/Zhou/dataset/UAV-Radar"
target_root = "/home/jackychou/dataset/UAV-Radar"

ra_real_mean = 4.753202145878351e-16
ra_real_std = 999.0759887695312
ra_imag_mean = -1.0385475201449713e-15
ra_imag_std = 999.0507202148438
rd_real_mean = -0.45920154452323914
rd_real_std = 1493.1441650390625
rd_imag_mean = -0.5187997221946716
rd_imag_std = 1491.2840576171875
ad_real_mean = -0.5542052984237671
ad_real_std = 459.1612243652344
ad_imag_mean = 3.830509662628174
ad_imag_std = 459.1851501464844

for idx in tqdm(range(1, 79  + 1)):
    source_path = os.path.join(source_root, f"uav_seqs_{idx}")
    target_path = os.path.join(target_root, f"uav_seqs_{idx}")
    check_path(target_path)
    source_path = os.path.join(source_path, "python_slice_frame_11")
    target_path = os.path.join(target_path, "python_slice_frame_11")
    check_path(target_path)
    source_path = os.path.join(source_path, "azimuth")
    target_path = os.path.join(target_path, "azimuth")
    check_path(target_path)

    # ra
    sub_source_path = os.path.join(source_path, "raw_frame_RA")
    sub_target_path = os.path.join(target_path, "raw_frame_RA")
    check_path(sub_target_path)

    for frame_idx in range(0, 300):
        for chirp_idx in range(1, 256):
            source_name = os.path.join(sub_source_path, f"{frame_idx:03d}_{chirp_idx:09d}.npy")
            target_name = os.path.join(sub_target_path, f"{frame_idx:03d}_{chirp_idx:09d}.npy")
            sub_file = np.load(source_name)
            sub_file = sub_file.astype(np.float64, copy=False)
            sub_file[..., 0] = (sub_file[..., 0] - ra_real_mean) / ra_real_std
            sub_file[..., 1] = (sub_file[..., 1] - ra_imag_mean) / ra_imag_std
            sub_file = sub_file.astype(np.float32, copy=False)
            np.save(target_name, sub_file)

    # rd
    sub_source_path = os.path.join(source_path, "raw_frame_RD")
    sub_target_path = os.path.join(target_path, "raw_frame_RD")
    check_path(sub_target_path)
    for frame_idx in range(0, 300):
        source_name = os.path.join(sub_source_path, f"{frame_idx:09d}.npy")
        target_name = os.path.join(sub_target_path, f"{frame_idx:09d}.npy")
        sub_file = np.load(source_name)
        sub_file = sub_file.astype(np.float64, copy=False)
        sub_file[..., 0] = (sub_file[..., 0] - rd_real_mean) / rd_real_std
        sub_file[..., 1] = (sub_file[..., 1] - rd_imag_mean) / rd_imag_std
        sub_file = sub_file.astype(np.float32, copy=False)
        np.save(target_name, sub_file)

    # ad
    sub_source_path = os.path.join(source_path, "raw_frame_AD")
    sub_target_path = os.path.join(target_path, "raw_frame_AD")
    check_path(sub_target_path)
    for frame_idx in range(0, 300):
        source_name = os.path.join(sub_source_path, f"{frame_idx:09d}.npy")
        target_name = os.path.join(sub_target_path, f"{frame_idx:09d}.npy")
        sub_file = np.load(source_name)
        sub_file = sub_file.astype(np.float64, copy=False)
        sub_file[..., 0] = (sub_file[..., 0] - ad_real_mean) / ad_real_std
        sub_file[..., 1] = (sub_file[..., 1] - ad_imag_mean) / ad_imag_std
        sub_file = sub_file.astype(np.float32, copy=False)
        np.save(target_name, sub_file)