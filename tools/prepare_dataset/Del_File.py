import os
import subprocess
from tqdm import tqdm



def merge_creat_new(x):
    new = [item for sublist in x for item in sublist]
    out = []
    for i in new:
        if i not in out:
            out.append(i)
    return out

root = "/home/jackychou/dataset/UAVRadar"

save_chirps = [
    [1, 85, 170, 255],
    [1, 37, 73, 109, 146, 182, 218, 255],
    [1, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255],
    [1, 9, 17, 25, 33, 41, 50, 58, 66, 74, 82, 91, 99, 107, 115, 123, 132, 140, 148, 156, 164, 173, 181, 189, 197, 205,
     214, 222, 230, 238, 246, 255],
    [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109,
     113, 117, 121, 125, 130, 134, 138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186, 190, 194, 198, 202,
     206, 210, 214, 218, 222, 226, 230, 234, 238, 242, 246, 250, 255, 128],
    # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59,
    #  61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113,
    #  115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159,
    #  161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197, 199, 201, 203, 205,
    #  207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251,
    #  253, 255],
]

save_chirps = merge_creat_new(save_chirps)
Idx_list = [i for i in range(1, 79, + 1)]

for idx in tqdm(Idx_list):
    # filter_types = [11]
    # for filter_type in filter_types:
    #     FilePath = os.path.join(root, f"uav_seqs_{idx}", f"python_slice_frame_{filter_type}", "azimuth", 'raw_frame_AD')
    #     param = f'rm -r {FilePath}'
    #     subprocess.run(param, shell=True)
    #
    #     FilePath = os.path.join(root, f"uav_seqs_{idx}", f"python_slice_frame_{filter_type}", "azimuth", 'raw_frame_RD')
    #     param = f'rm -r {FilePath}'
    #     subprocess.run(param, shell=True)

    # FilePath = os.path.join(root, f"uav_seqs_{idx}", "annot")
    # param = f'rm -r {FilePath}'
    # subprocess.run(param, shell=True)

    # FilePath = os.path.join(root, f"uav_seqs_{idx}", "python_slice_frame_11_PCA")
    # param = f'rm -r {FilePath}'
    # print(param)
    # subprocess.run(param, shell=True)

    # FilePath = os.path.join(root, f"uav_seqs_{idx}", "python_slice_frame_11_PCAv2")
    # param = f'rm -r {FilePath}'
    # print(param)
    # subprocess.run(param, shell=True)

    # FilePath = os.path.join(root, f"uav_seqs_{idx}", "adc_interval")
    # param = f'rm -r {FilePath}'
    # subprocess.run(param, shell=True)

    # FilePath = os.path.join(root, f"uav_seqs_{idx}", f"python_slice_frame_{filter_type}", 'azimuth', "raw_frame_AD")
    # param = f'rm -r {FilePath}'
    # subprocess.run(param, shell=True)
    #
    # FilePath = os.path.join(root, f"uav_seqs_{idx}", f"python_slice_frame_{filter_type}", 'azimuth', "raw_frame_RD")
    # param = f'rm -r {FilePath}'
    # subprocess.run(param, shell=True)

    filter_type = 11
    for frame_idx in range(0, 300):
        for chirp_idx in range(1, 256):
            if chirp_idx not in save_chirps:
                FilePath = os.path.join(root, f"uav_seqs_{idx}", f"python_slice_frame_{filter_type}", 'azimuth', "raw_frame_RA", f"{frame_idx:03d}_{chirp_idx:09d}.npy")
                if os.path.exists(FilePath):
                    param = f'rm {FilePath}'
                    subprocess.run(param, shell=True)

    # FilePath = os.path.join(root, f"uav_seqs_{idx}", "python_slice_frame_1", "azimuth", "raw_frame_RD")
    # param = f'rm -r {FilePath}'
    # subprocess.run(param, shell=True)

    # FilePath = os.path.join(root, f"uav_seqs_{idx}", "csv_offset_label_rad_1")
    # param = f'rm -r {FilePath}'
    # subprocess.run(param