import os
import subprocess
import time



def main():
    # 修改 Start
    config_list = [
        # 'paperv1/stct/rodnet-mnet-single-32_128_1132_baseline_r&i.py',
        # 'paperv1/backbone/rodnet-STCT_cdc-single-32_128_1132_baseline_r&i.py',
        'paperv1/loss/rodnet-STCTNet_bcel-single-32_128_1132_baseline_r&i.py',
        # 'paperv1/loss/rodnet-STCTNet_gaussian_0_1-single-32_128_1132_baseline_r&i.py',
        # 'CE/rodnet-CDC-single-32_128_11_baseline_r&i.py',
        ]

    sensor_list = [
        # 'pca_32.json',
        # 'pca_32.json',
        'pca_32.json',
        # 'pca_32.json',
        # 'uniform_32.json',
    ]

    data_list = [
        # 'train_test_32_11_128_PCAv1',
        # 'train_test_32_11_128_PCAv1',
        'train_test_32_11_128_PCAv1',
        # 'train_test_32_11_128_PCAv1',
        # 'train_test_32_11_128',
    ]

    checkpoint_list = [
        # 'paperv1/stct/mnet/myNet-20260308-060300/epoch_19_best.pkl',
        # 'paperv1/backbone/cdc/CDC+STCT-20260303-223220/epoch_19_best.pkl',
        'paperv1/loss/STCT_PCA_bcel/myNet-20260320-214905/epoch_20_best.pkl',
        # 'paperv1/pca/32/myNet-20260228-153128/epoch_15_best.pkl',
        # 'CE/CDC/CDC+MNet-20260129-171452/epoch_19_best.pkl',
    ]

    save_list = [
        # 'wo_STCT',
        # 'wo_backbone',
        'wo_softBCE',
        # 'full',
        # 'cdc',
    ]
    # 修改 End

    python_path = '/home/jackychou/.conda/envs/UAVRadar/bin/python3'
    forward_path = '/home/jackychou/Zhou/code/UAVRadar/tools/paperv1/model_confmap.py'
    config_root = '/home/jackychou/Zhou/code/UAVRadar/configs'
    data_root = '/home/jackychou/dataset/UAVRadar'
    sensor_root = '/home/jackychou/Zhou/code/UAVRadar/cruw-devkit/cruw/dataset_configs'
    checkpoint_root = '/home/jackychou/Zhou/code/UAVRadar/logs'
    save_root = '/mnt/d/paperv1/labels'

    for config_path, sensor_path, data_path, checkpoint_path, save_path in zip(config_list, sensor_list, data_list, checkpoint_list, save_list):
        param = f'"{python_path}" "{forward_path}" --config "{os.path.join(config_root, config_path)}" --sensor_config "{os.path.join(sensor_root, sensor_path)}" --data_dir "{os.path.join(data_root, data_path)}" --checkpoint "{os.path.join(checkpoint_root, checkpoint_path)}" --res_dir "{os.path.join(save_root, save_path)}"'
        subprocess.run(param, shell=True)
        time.sleep(60)

if __name__ == '__main__':
    main()