import os, sys, inspect
import scipy
import math
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import pandas as pd
# from configs.UAV_60_radar import radar_config
from configs.rudet import radar_configs
import ast
import numpy as np



release_dataset_label_map = {0: 'uav'}
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def confmap2ra(radar_configs, name, radordeg='rad'):
    Fs = radar_configs['sample_freq']
    sweepSlope = radar_configs['sweep_slope']
    num_crop = radar_configs['crop_num']
    fft_Rang = radar_configs['ramap_rsize'] + 2 * num_crop
    fft_Ang = radar_configs['ramap_asize']
    c = scipy.constants.speed_of_light

    if name == 'range':
        freq_res = Fs / fft_Rang
        freq_grid = np.arange(fft_Rang) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2
        rng_grid = rng_grid[num_crop:fft_Rang - num_crop]
        return rng_grid

    if name == 'angle':
        # for [-90, 90], w will be [-1, 1]
        w = np.linspace(math.sin(math.radians(radar_configs['ra_min'])),
                        math.sin(math.radians(radar_configs['ra_max'])),
                        radar_configs['ramap_asize'])
        if radordeg == 'deg':
            agl_grid = np.degrees(np.arcsin(w))  # rad to deg
        elif radordeg == 'rad':
            agl_grid = np.arcsin(w)
        else:
            raise TypeError
        return agl_grid


def labelmap2ra(radar_configs, name, radordeg='rad'):
    Fs = radar_configs['sample_freq']
    sweepSlope = radar_configs['sweep_slope']
    num_crop = radar_configs['crop_num']
    fft_Rang = radar_configs['ramap_rsize_label'] + 2 * num_crop
    fft_Ang = radar_configs['ramap_asize_label']
    c = scipy.constants.speed_of_light

    if name == 'range':
        freq_res = Fs / fft_Rang
        freq_grid = np.arange(fft_Rang) * freq_res
        rng_grid = freq_grid * c / sweepSlope / 2
        rng_grid = rng_grid[num_crop:fft_Rang - num_crop]
        # rng_grid = np.flip(rng_grid)
        return rng_grid

    if name == 'angle':
        if radordeg == 'rad':
            agl_grid = np.linspace(math.radians(radar_configs['ra_min_label']),
                                   math.radians(radar_configs['ra_max_label']),
                                   radar_configs['ramap_asize_label'])  # deg to rad
        elif radordeg == 'deg':
            agl_grid = np.linspace(radar_configs['ra_min_label'], radar_configs['ra_max_label'],
                                   radar_configs['ramap_asize_label'])  # keep deg
        else:
            raise TypeError
        return agl_grid


def convert(seq_path, single_id, adc_interval_len):
    seq_path = os.path.join(seq_path, f"uav_seqs_{single_id}")
    if os.path.exists(os.path.join(seq_path, "annot")) is False:
        os.mkdir(os.path.join(seq_path, "annot"))
    adc_interval_path = os.path.join(seq_path, f"adc_interval/new_interval_{adc_interval_len}.txt")

    range_grid = labelmap2ra(radar_configs, name='range')
    angle_grid = labelmap2ra(radar_configs, name='angle')

    images_path = os.path.join(seq_path, "camera_to_frame")
    label_path = os.path.join(seq_path, f"csv_offset_label_{label_type}")

    if os.path.exists(label_path) is False:
        label_path = os.path.join(seq_path, f"csv_offset_label_{label_type}")

    files = sorted(os.listdir(label_path))

    file_attributes = 'rtk'
    region_shape_attributes = {"name": "point", "cx": 0, "cy": 0}
    region_attributes = {"class": None}
    columns = ['filename', 'file_size', 'file_attributes', 'region_count', 'region_id', 'region_shape_attributes', 'region_attributes']
    data = []

    for file in files:
        file_dir = os.path.join(label_path, file)
        label = open(file_dir)
        img_name = file.replace("csv", "jpg")
        img_size = os.path.getsize(os.path.join(images_path, img_name))
        region_count = 0
        obj_info = []

        # parse a label file
        for line in label:
            line = line.rstrip().split(',')
            if int(line[1]) in release_dataset_label_map:
                type_ = release_dataset_label_map[int(line[1])]
            else:
                continue

            angle = float(line[2])
            distance = float(line[3])

            rng_idx, _ = find_nearest(range_grid, distance)
            agl_idx, _ = find_nearest(angle_grid, angle)

            # TODO
            if label_type == "rad":
                if use_crop:
                    with open(adc_interval_path, "r") as file:
                        content = file.read()
                        new_adc_interval = ast.literal_eval(content.split('\n')[-1])
                    new_rng_idx = int(rng_idx - new_adc_interval[0])
                    assert new_rng_idx >= 0, f"新区间不包含该帧的距离索引."
                    obj_info.append([range_grid[new_rng_idx], angle, type_])
                else:
                    obj_info.append([distance, angle, type_])
            elif label_type == "deg":
                if use_crop:
                    with open(adc_interval_path, "r") as file:
                        content = file.read()
                        new_adc_interval = ast.literal_eval(content.split('\n')[-1])
                    new_rng_idx = int(rng_idx - new_adc_interval[0])
                    assert new_rng_idx >= 0, f"新区间不包含该帧的距离索引."
                    obj_info.append([new_rng_idx, agl_idx, type_])    # RAMP+CropADC
                else:
                    obj_info.append([rng_idx, agl_idx, type_])    # RAMP
            region_count += 1

        for objId, obj in enumerate(obj_info):  # set up rows for different objs
            row = []
            row.append(img_name)
            row.append(img_size)
            row.append(file_attributes)
            row.append(region_count)
            row.append(objId)

            if label_type == "rad":
                region_shape_attributes["cx"] = float(obj[1])  # float --> RODNet
                region_shape_attributes["cy"] = float(obj[0])  # float --> RODNet
            elif label_type == "deg":
                region_shape_attributes["cx"] = int(obj[1])   # float --> RODNet
                region_shape_attributes["cy"] = int(obj[0])   # float --> RODNet
            if int(obj[0]) == 1:
                print(obj)
            region_attributes["class"] = obj[2]
            row.append(json.dumps(region_shape_attributes))
            row.append(json.dumps(region_attributes))
            data.append(row)

    df = pd.DataFrame(data, columns=columns)

    df.to_csv(os.path.join(os.path.join(seq_path, "annot"), f"rodnet_labels_{adc_interval_len}_{label_type}.csv"), index=None, header=True)

    print("\tSuccess!")


if __name__ == "__main__":
    base_root = '/home/jackychou/Zhou/dataset/UAVRadar'
    start_single_id = 1
    end_single_id = 79
    use_crop = True
    label_type = "rad" # rad
    adc_interval_lens = [128, 256, 512]  # 256

    for adc_interval_len in adc_interval_lens:
        for single_id in range(start_single_id, end_single_id + 1):
            convert(base_root, single_id, adc_interval_len)

