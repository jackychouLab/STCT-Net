import os



source = "/mnt/extra_data/ZJH/exoUAV1.0"
target = "/mnt/extra_data/ZJH/UAV1.0"
start_idx = 1
end_idx = 79
# sub_names = [114, 118, 1116, 1132, 1164]

for idx in range(start_idx, end_idx+1):
    # for sub_name in sub_names:
    #     src_name = os.path.join(root, f"uav_seqs_{idx}", f"PCA_slice_frame_{sub_name}")
    #     tar_name = os.path.join(root, f"uav_seqs_{idx}", f"python_slice_frame_{sub_name}")
    #     os.rename(src_name, tar_name)
    s_path = os.path.join(source, f"uav_seqs_{idx}", "python_slice_frame_11")
    t_path = os.path.join(target, f"uav_seqs_{idx}")
    os.system(f'mv "{s_path}" "{t_path}"')
