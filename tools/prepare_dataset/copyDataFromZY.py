import os
from tqdm import tqdm



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


source_root = "/mnt/c/Ubuntu-temp/mmUAV"
target_root = "/home/jackychou/Zhou/dataset/UAV1.0"
start_idx = 1
end_idx = 79

for idx in tqdm(range(start_idx, end_idx + 1)):
    source_path = os.path.join(source_root, f"uav_seqs_{idx}")
    target_path = os.path.join(target_root, f"uav_seqs_{idx}")
    check_path(target_path)
    # label raw_data
    # os.system(f'cp -r "{os.path.join(source_path, "adc_interval")}" "{target_path}"')
    # os.system(f'cp -r "{os.path.join(source_path, "raw_radar")}" "{target_path}"')
    # video
    # os.system(f'cp -r "{os.path.join(source_path, "camera_video")}" "{target_path}"')
    # os.system(f'cp -r "{os.path.join(source_path, "camera_to_frame")}" "{target_path}"')
    os.system(f'cp -r "{os.path.join(source_path, "RTK_Label_128")}" "{target_path}"')
