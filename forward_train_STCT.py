import os
import subprocess
import pynvml
import time
import gc
import torch


def get_gpu_info(gpu_num):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_num)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.free / 1024 / 1024 / 1024


def replace_norm_type(file_path, old_base, new_base):
    with open(file_path, 'r', encoding='utf-8') as f:
        new_content = f.read()

    # 替换 norm_type 的值
    for old, new in zip(old_base, new_base):
        new_content = new_content.replace(old, new)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



def main():
    # 修改 Start
    config_path = 'rodnet-myNet-single-32_128_1132_baseline_r&i'
    base_mnet = "mnet_type=0"
    base_norm = "norm_type='0'"
    base_act = "act_type='0'"
    base_conv = "full_conv=0"
    base_loss = "loss_type='0'"
    base_train = "train_type='0'"
    old_base_list = [base_mnet, base_norm, base_act, base_conv, base_loss, base_train]
    sub_config_list = [
        # ["mnet_type=5", "norm_type='gn'", "act_type='gelu'", "full_conv=True", "loss_type='bce'", "train_type='multi'"],
        # ["mnet_type=5", "norm_type='gn'", "act_type='gelu'", "full_conv=True", "loss_type='mse'", "train_type='multi'"],
        ["mnet_type=5", "norm_type='gn'", "act_type='gelu'", "full_conv=True", "loss_type='smooth_l1'", "train_type='multi'"],
        # ["mnet_type=5", "norm_type='gn'", "act_type='gelu'", "full_conv=True", "loss_type='mae'", "train_type='multi'"],
        ]
    # 修改 End

    python_path = '/home/jackychou/.conda/envs/paperv1/bin/python3'
    train_path = '/home/jackychou/code/RODNet_UAV/tools/train.py'
    config_root = '/home/jackychou/code/RODNet_UAV/configs'
    data_root = '/home/jackychou/dataset/UAV1.0'
    sensor_root = '/home/jackychou/code/RODNet_UAV/cruw-devkit/cruw/dataset_configs'
    log_root = '/home/jackychou/code/RODNet_UAV/workers'

    while(True):
        if get_gpu_info(0)  >= 20:
            break
        time.sleep(120)

    chirp_idx = config_path.split('-')[-1].split('_')[0]
    crop_len = config_path.split('-')[-1].split('_')[1]
    use_filter = config_path.split('-')[-1].split('_')[2]
    data_path = f"train_test_{chirp_idx}_{use_filter}_{crop_len}"
    if int(use_filter) >= 110:
        sensor_path = f"pca_{chirp_idx}"
    else:
        sensor_path = f"uniform_{chirp_idx}"

    for new_base_list in sub_config_list:
        cleanup()

        config_abs_path = os.path.join(config_root, config_path + ".py")
        replace_norm_type(config_abs_path, old_base_list, new_base_list)
        old_base_list = new_base_list
        with open(config_abs_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(content)
        param = f'"{python_path}" "{train_path}" --config "{os.path.join(config_root, config_path + ".py")}" --sensor_config "{os.path.join(sensor_root, sensor_path + ".json")}" --data_dir "{os.path.join(data_root, data_path)}" --log_dir "{os.path.join(log_root, config_path)}"'
        subprocess.run(param, shell=True)

        time.sleep(60)

if __name__ == '__main__':
    main()