import os
import subprocess
import time



def main():
    # Note: Start
    config_list = [
        'UAVRadar/CE-PCA/CDC-DCN-PCA',
        ]
    code_root = '/home/jackychou/Zhou/code/STCT-Net'
    python_path = '/home/jackychou/software/miniconda3/envs/STCT-Net/bin/python3'
    data_root = '/home/jackychou/dataset/UAVRadar'
    # End
    config_root = os.path.join(code_root, 'configs')
    sensor_root = os.path.join(config_root, 'UAVRadar/dataset_configs')
    log_root = os.path.join(code_root, 'logs')
    train_path = os.path.join(code_root, 'tools/train_UAVRadar.py')

    if os.path.exists(log_root) is False:
        os.mkdir(log_root)

    for config_path in config_list:
        if 'PCA' in config_path:
            data_path = 'train_test_32_11_128_PCAv1'
        else:
            data_path = 'train_test_32_11_128'
        chirp_idx = data_path.split('_')[2]
        if 'PCA' in data_path or 'pca' in data_path:
            sensor_path = f"pca_{chirp_idx}"
        else:
            sensor_path = f"uniform_{chirp_idx}"

        param = f'PYTHONPATH=. "{python_path}" "{train_path}" --config "{os.path.join(config_root, config_path + ".py")}" --sensor_config "{os.path.join(sensor_root, sensor_path + ".json")}" --data_dir "{os.path.join(data_root, data_path)}" --log_dir "{os.path.join(log_root, config_path)}" --code_dir "{code_root}"'
        subprocess.run(param, shell=True)
        time.sleep(60)


if __name__ == '__main__':
    main()