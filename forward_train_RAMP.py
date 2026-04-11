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
        # 'CE/rodnet-RAMP-single-32_128_11_baseline_r&i1',
        'CE/rodnet-RAMP-single-32_128_11_baseline_r&i2',
        # 'CE/rodnet-RAMP-single-32_128_11_baseline_r&i3',
        ]

    data_list = [
        # 'train_test_32_11_128',
        'train_test_32_11_128',
        # 'train_test_32_11_128',
    ]

    log_list = [
        # 'CE/rampv1',
        'CE/rampv2',
        # 'CE/rampv3',
    ]
    # 修改 End

    python_path = '/home/jackychou/.conda/envs/UAVRadar/bin/python3'
    train_path = '/home/jackychou/Zhou/code/UAVRadar/tools/train_ramp.py'
    config_root = '/home/jackychou/Zhou/code/UAVRadar/configs'
    data_root = '/home/jackychou/dataset/UAVRadar'
    sensor_root = '/home/jackychou/Zhou/code/UAVRadar/cruw-devkit/cruw/dataset_configs'
    log_root = '/home/jackychou/Zhou/code/UAVRadar/logs'

    while(True):
        if get_gpu_info(0)  >= 20:
            break
        time.sleep(120)

    for config_path, data_path, log_path in zip(config_list, data_list, log_list):
        chirp_idx = config_path.split('-')[-1].split('_')[0]

        if 'PCA' in data_path or 'pca' in data_path:
            sensor_path = f"pca_{chirp_idx}"
        else:
            sensor_path = f"uniform_{chirp_idx}"

        param = f'"{python_path}" "{train_path}" --config "{os.path.join(config_root, config_path + ".py")}" --sensor_config "{os.path.join(sensor_root, sensor_path + ".json")}" --data_dir "{os.path.join(data_root, data_path)}" --log_dir "{os.path.join(log_root, log_path)}"'
        subprocess.run(param, shell=True)
        time.sleep(60)

if __name__ == '__main__':
    main()