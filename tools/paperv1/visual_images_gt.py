import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cruw.mapping import ra2idx
import math
from configs.UAV_60_radar import radar_config
import ast
from cruw.annotation.init_json import init_meta_json
import json



def normalize_confmap(confmap):
    conf_min = np.min(confmap)
    conf_max = np.max(confmap)
    if conf_max - conf_min != 0:
        confmap_norm = (confmap - conf_min) / (conf_max - conf_min)
    else:
        confmap_norm = confmap
    return confmap_norm


def generate_confmap(rng_idx, agl_idx, gaussian_thres=36):
    confmap_cfg = dict(
        confmap_sigmas={
            'uav': 10,  # Z. Jiang:10     J. Zhou:5
        },
        confmap_sigmas_interval={
            'uav': [5, 10],  # Z. Jiang:[5, 10]   J. Zhou:[3, 5]
        },
        confmap_length={
            'uav': 1,
        }
    )

    confmap_sigmas = confmap_cfg['confmap_sigmas']
    confmap_sigmas_interval = confmap_cfg['confmap_sigmas_interval']
    confmap_length = confmap_cfg['confmap_length']

    range_grid = radar_config['rng_grid'][radar_config['num_crop']: -radar_config['num_crop']]

    confmap = np.zeros((1, 128, 128), dtype=float)

    class_name = 'uav'

    class_id = 0
    sigma = 2 * np.arctan(confmap_length[class_name] / (2 * range_grid[rng_idx])) * confmap_sigmas[class_name]
    sigma_interval = confmap_sigmas_interval[class_name]
    if sigma > sigma_interval[1]:
        sigma = sigma_interval[1]
    if sigma < sigma_interval[0]:
        sigma = sigma_interval[0]
    for i in range(128):
        for j in range(128):
            distant = (((rng_idx - i) * 2) ** 2 + (agl_idx - j) ** 2) / sigma ** 2
            if distant < gaussian_thres:  # threshold for confidence maps
                value = np.exp(- distant / 2) / (2 * math.pi)
                confmap[class_id, i, j] = value if value > confmap[class_id, i, j] else confmap[class_id, i, j]

    return confmap


def load_anno_csv(csv_path, n_frame):

    file_type = "python"
    folder_name_dict = dict(
        cam_0='camera_to_frame',
        rad_h=f'{file_type}_slice_frame_{use_filter}'
    )
    anno_dict = init_meta_json(n_frame, folder_name_dict)
    data = pd.read_csv(csv_path)
    n_row, n_col = data.shape

    for r in range(n_row):
        filename = data['filename'][r]
        frame_id = int(filename.split('.')[0].split('_')[-1])
        region_count = data['region_count'][r]

        if region_count != 0:
            region_shape_attri = json.loads(data['region_shape_attributes'][r])
            region_attri = json.loads(data['region_attributes'][r])
            cx = region_shape_attri['cx']
            cy = region_shape_attri['cy']
            class_name = region_attri['class']

            rid, aid = ra2idx(cy, cx, radar_config['rng_grid'], radar_config['agl_grid'])

            anno_dict[frame_id]['rad_h']['n_objects'] += 1
            anno_dict[frame_id]['rad_h']['obj_info']['categories'].append(class_name)
            anno_dict[frame_id]['rad_h']['obj_info']['centers'].append([cy, cx])
            anno_dict[frame_id]['rad_h']['obj_info']['center_ids'].append([rid, aid])
            anno_dict[frame_id]['rad_h']['obj_info']['scores'].append(1.0)

    return anno_dict


data_root = "/home/jackychou/Zhou/dataset/UAVRadar"
save_root = "/mnt/d/paperv1/labels"
idx_list = [2, 6, 7, 8, 37, 45, 47, 55, 59, 63, 65, 71, 73, 74, 77]
# idx_list = [71]
use_filter = 11
frames_list = [i for i in range(0, 300)]
select_chirps = [i for i in range(1, 256)]

for seq_idx in idx_list:
    data_type = "python"

    # save RA
    sub_path = os.path.join(data_root, f"uav_seqs_{str(seq_idx)}", f"{data_type}_slice_frame_{str(use_filter)}", "azimuth")
    save_ra_path = os.path.join(save_root, 'ra', f"uav_seqs_{str(seq_idx)}")
    if os.path.exists(save_ra_path) is False:
        os.makedirs(save_ra_path)
    RA_files = os.path.join(sub_path, "raw_frame_RA")
    adc_interval_path = os.path.join(data_root, f"uav_seqs_{str(seq_idx)}", f"adc_interval/new_interval_128.txt")
    with open(adc_interval_path, "r") as file:
        content = file.read()
        adc_interval = ast.literal_eval(content.split('\n')[-1])
    for frame_idx in frames_list:
        RA = None
        for select_chirp in select_chirps:
            sub_RA = np.load(os.path.join(RA_files, f"{frame_idx:03d}_{select_chirp:09d}.npy"))
            sub_RA = sub_RA[:, :, 0] + 1j * sub_RA[:, :, 1]
            sub_RA = np.abs(sub_RA[adc_interval[0]:adc_interval[1] + 1, ...])
            break
            if RA is None:
                RA = sub_RA
            else:
                RA += sub_RA

        plt.close('all')
        fig = plt.figure()
        plt.imshow(sub_RA, origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        sub_save_path = os.path.join(save_ra_path, f"{frame_idx:09d}.jpg")
        plt.savefig(sub_save_path, dpi=300, bbox_inches='tight', pad_inches=0)

    # save ConfMap
    save_confmap_path = os.path.join(save_root, 'confmap', f"uav_seqs_{str(seq_idx)}")
    if os.path.exists(save_confmap_path) is False:
        os.mkdir(save_confmap_path)

    csv_path = f'/home/jackychou/dataset/UAVRadar/uav_seqs_{seq_idx}/annot/rodnet_labels_128_rad.csv'
    gts = load_anno_csv(csv_path, 300)

    for frame_idx, metadata_frame in enumerate(gts):
        n_obj = metadata_frame['rad_h']['n_objects']
        obj_info = metadata_frame['rad_h']['obj_info']
        rng_idx = obj_info['center_ids'][0][0]
        agl_idx = obj_info['center_ids'][0][1]

        sub_confmap_gt = generate_confmap(rng_idx, agl_idx)
        sub_confmap_gt = normalize_confmap(sub_confmap_gt)
        sub_confmap_gt = sub_confmap_gt[0, :, :]
        plt.close('all')
        fig = plt.figure()
        plt.imshow(sub_confmap_gt, origin='lower')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        sub_save_path = os.path.join(save_confmap_path, f"{frame_idx:09d}.jpg")
        plt.savefig(sub_save_path, dpi=300, bbox_inches='tight', pad_inches=0)


        # save Images
        save_image_path = os.path.join(save_root, 'image', f"uav_seqs_{str(seq_idx)}")
        if os.path.exists(save_image_path) is False:
            os.mkdir(save_image_path)
        image_source_path = os.path.join(data_root, f"uav_seqs_{str(seq_idx)}", 'camera_to_frame', f"{frame_idx:09d}.jpg")
        os.system(f'cp "{image_source_path}" "{save_image_path}"/')