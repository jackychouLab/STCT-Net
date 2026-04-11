import os
import subprocess
import pynvml
import time


def get_gpu_info(gpu_num):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_num)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return meminfo.free / 1024 / 1024 / 1024


def main():
    # 修改 Start
    config_list = [
        'rodnet-DCSN-single-4_128_11_baseline_r&i',
        ]
    # 修改 End

    python_path = '/home/jackychou/.conda/envs/paperv1/bin/python3'
    train_path = '/home/jackychou/code/RODNet_UAV/tools/train_dcsn.py'
    config_root = '/home/jackychou/code/RODNet_UAV/configs'
    data_root = '/home/jackychou/dataset/UAV1.0'
    sensor_root = '/home/jackychou/code/RODNet_UAV/cruw-devkit/cruw/dataset_configs'
    log_root = '/home/jackychou/code/RODNet_UAV/workers'

    while(True):
        if get_gpu_info(0)  >= 20:
            break
        time.sleep(120)

    for config_path in config_list:
        chirp_idx = config_path.split('-')[-1].split('_')[0]
        crop_len = config_path.split('-')[-1].split('_')[1]
        use_filter = config_path.split('-')[-1].split('_')[2]
        data_path = f"train_test_{chirp_idx}_{use_filter}_{crop_len}"
        if int(use_filter) >= 110:
            sensor_path = f"pca_{chirp_idx}"
        else:
            sensor_path = f"uniform_{chirp_idx}"

        param = f'"{python_path}" "{train_path}" --config "{os.path.join(config_root, config_path + ".py")}" --sensor_config "{os.path.join(sensor_root, sensor_path + ".json")}" --data_dir "{os.path.join(data_root, data_path)}" --log_dir "{os.path.join(log_root, config_path)}"'
        subprocess.run(param, shell=True)
        time.sleep(60)

if __name__ == '__main__':
    main()