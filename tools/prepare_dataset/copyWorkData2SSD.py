import os
import subprocess
from tqdm import tqdm
import shutil
import time



def merge_creat_new(x):
    new = [item for sublist in x for item in sublist]
    out = []
    for i in new:
        if i not in out:
            out.append(i)
    return out


def check_path(x):
    if type(x) == str:
        x = [x]
    for sub_x in x:
        if os.path.exists(sub_x) is False:
            os.mkdir(sub_x)


chirp_list = [
    # [1, 85, 170, 255],
    # [1, 37, 73, 109, 146, 182, 218, 255],
    # [1, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255],
    [1, 9, 17, 25, 33, 41, 50, 58, 66, 74, 82, 91, 99, 107, 115, 123, 132, 140, 148, 156, 164, 173, 181, 189, 197, 205,
     214, 222, 230, 238, 246, 255, 128],
    # [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109,
    #  113, 117, 121, 125, 130, 134, 138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186, 190, 194, 198, 202,
    #  206, 210, 214, 218, 222, 226, 230, 234, 238, 242, 246, 250, 255, 128],
    # [i for i in range(1, 64 + 1)]
]

source_root = "/home/jackychou/dataset/UAVRadar"
target_root = "/home/jackychou/dataset/UAVRadar"
check_path(target_root)

start_idx = 1
end_idx = 79
chirp_list = merge_creat_new(chirp_list)
PCA_num = 32
Idxlist = [i for i in range(start_idx, end_idx + 1)]


for idx in tqdm(Idxlist):
    source_path = os.path.join(source_root, f"uav_seqs_{idx}")
    target_path = os.path.join(target_root, f"uav_seqs_{idx}")
    check_path(target_path)

    ### base files
    # os.system(f'cp -r "{os.path.join(source_path, "adc_interval")}" "{target_path}"')
    # os.system(f'cp -r "{os.path.join(source_path, "camera_to_frame")}" "{target_path}"')
    # os.system(f'cp -r "{os.path.join(source_path, "annot")}" "{target_path}"')
    # os.system(f'cp -r "{os.path.join(source_path, "csv_offset_label_rad")}" "{target_path}"')

    ### exo files
    # os.system(f'cp -r "{os.path.join(source_path, "raw_radar")}" "{target_path}"')

    ### main files 11
    frame_types = [11]
    for frame_type in frame_types:
        check_path([os.path.join(target_path, f"python_slice_frame_{frame_type}"), os.path.join(target_path, f"python_slice_frame_{frame_type}", "azimuth")])
        sub_source_path = os.path.join(source_path, f"python_slice_frame_{frame_type}", "azimuth")
        sub_target_path = os.path.join(target_path, f"python_slice_frame_{frame_type}", "azimuth")

        os.system(f'cp -r "{os.path.join(sub_source_path, "raw_frame_AD")}" "{sub_target_path}"')
        os.system(f'cp -r "{os.path.join(sub_source_path, "raw_frame_RD")}" "{sub_target_path}"')
        # os.system(f'cp -r "{os.path.join(sub_source_path, "raw_frame_RA")}" "{sub_target_path}"')

        # n_frame = 300
        # check_path(os.path.join(sub_target_path, "raw_frame_RA"))
        # for frame_idx in range(n_frame):
        #     for chirp_idx in chirp_list:
        #         if os.path.exists(os.path.join(sub_target_path, "raw_frame_RA", f"{frame_idx:03d}_{chirp_idx:09d}.npy")) is False:
        #             os.system(f'cp "{os.path.join(sub_source_path, "raw_frame_RA", f"{frame_idx:03d}_{chirp_idx:09d}.npy")}" "{os.path.join(sub_target_path, "raw_frame_RA", f"{frame_idx:03d}_{chirp_idx:09d}.npy")}"')

    # main files pca11
    chirp_list = [i for i in range(1, PCA_num + 1)]
    # for frame_type in frame_types:
    #     check_path([os.path.join(target_path, f"python_slice_frame_{frame_type}_PCAv1"), os.path.join(target_path, f"python_slice_frame_{frame_type}_PCAv1", "azimuth")])
    #     sub_source_path = os.path.join(source_path, f"python_slice_frame_{frame_type}_PCAv1", "azimuth")
    #     sub_target_path = os.path.join(target_path, f"python_slice_frame_{frame_type}_PCAv1", "azimuth")
    #
    #     n_frame = 300
    #     check_path(os.path.join(sub_target_path, "raw_frame_RA"))
    #     for frame_idx in range(n_frame):
    #         for chirp_idx in chirp_list:
    #             if os.path.exists(os.path.join(sub_target_path, "raw_frame_RA", f"{frame_idx:03d}_{chirp_idx:09d}.npy")) is False:
    #                 os.system(f'cp "{os.path.join(sub_source_path, "raw_frame_RA", f"{frame_idx:03d}_{chirp_idx:09d}.npy")}" "{os.path.join(sub_target_path, "raw_frame_RA", f"{frame_idx:03d}_{chirp_idx:09d}.npy")}"')

