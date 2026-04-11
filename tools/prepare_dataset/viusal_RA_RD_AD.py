import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from matplotlib.patches import Circle
from rodnet.core.object_class import get_class_id
import math
from configs.UAV_60_radar import radar_config



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
            'uav': 5,
        },
        confmap_sigmas_interval={
            'uav': [3, 5],
        },
        confmap_length={
            'uav': 1,
        }
    )

    confmap_sigmas = confmap_cfg['confmap_sigmas']
    confmap_sigmas_interval = confmap_cfg['confmap_sigmas_interval']
    confmap_length = confmap_cfg['confmap_length']

    range_grid = radar_config['rng_grid'][radar_config['num_crop']: -radar_config['num_crop']]

    confmap = np.zeros((1, 512, 128), dtype=float)

    class_name = 'uav'

    class_id = 0
    sigma = 2 * np.arctan(confmap_length[class_name] / (2 * range_grid[rng_idx])) * confmap_sigmas[class_name]
    sigma_interval = confmap_sigmas_interval[class_name]
    if sigma > sigma_interval[1]:
        sigma = sigma_interval[1]
    if sigma < sigma_interval[0]:
        sigma = sigma_interval[0]
    for i in range(512):
        for j in range(128):
            distant = (((rng_idx - i) * 2) ** 2 + (agl_idx - j) ** 2) / sigma ** 2
            if distant < gaussian_thres:  # threshold for confidence maps
                value = np.exp(- distant / 2) / (2 * math.pi)
                confmap[class_id, i, j] = value if value > confmap[class_id, i, j] else confmap[class_id, i, j]

    return confmap


def visual_ra_rd_ad_from_mat(ra, rd, ad, labels, save_path):
    plt.close()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 5)
    ax1 = fig.add_subplot(gs[0, 0])
    plt.imshow(ra, origin='lower')
    ax1.set_title("RA")

    ax2 = fig.add_subplot(gs[0, 1])
    plt.imshow(rd, origin='lower')
    ax2.set_title('RD')

    ax3 = fig.add_subplot(gs[0, 2])
    plt.imshow(ad, origin='lower')
    ax3.set_title('AD')

    a = 5
    b = -7
    ax4 = fig.add_subplot(gs[0, 3])
    plt.imshow(ra, origin='lower')
    circle = Circle((labels[0]+a, labels[1]+b), 1, color='r', fill=True)
    ax4.add_patch(circle)
    ax4.set_title('RA_labels')

    conf_map = generate_confmap(labels[0]+a, labels[1]+b, gaussian_thres=36)
    conf_map = normalize_confmap(conf_map)[0]

    ax5 = fig.add_subplot(gs[0, 4])
    plt.imshow(conf_map, origin='lower')
    ax5.set_title('confmap')

    plt.tight_layout()
    plt.savefig(save_path, dpi=600)

    plt.close()


def get_label_deg(csv_path):
    de = pd.read_csv(csv_path, delimiter="\t", header=None, skiprows=1)
    xs = []
    ys = []
    for index, row in de.iterrows():
        cx = int(row.to_dict()[0].split(', ""cy"": ')[0].split(" ")[-1])
        cy = int(row.to_dict()[0].split(', ""cy"": ')[1].split("}")[0])
        xs.append(cx)
        ys.append(cy)

    return xs, ys


data_root = "/home/jackychou/Zhou/dataset/UAV-Radar"
idx_list = [i for i in range(1, 1 + 1)]
use_filter = 11
frames_list = [i for i in range(0, 1)]
select_chirps = [i for i in range(1, 20+1)]

for seq_idx in idx_list:
    data_type = "python"
    sub_path = os.path.join(data_root, f"uav_seqs_{str(seq_idx)}", f"{data_type}_slice_frame_{str(use_filter)}_PCA", "azimuth")
    save_path = os.path.join(data_root, f"uav_seqs_{str(seq_idx)}", f"visual_{str(use_filter)}_PCA")
    # csv_path = os.path.join(data_root, f"uav_seqs_{str(seq_idx)}", "annot", "ramap_labels_512.csv")
    # xs, ys = get_label_deg(csv_path)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    RA_files = os.path.join(sub_path, "raw_frame_RA")
    RD_files = os.path.join(sub_path, "raw_frame_RD")
    AD_files = os.path.join(sub_path, "raw_frame_AD")

    for frame_idx in frames_list:
        # if frame_idx != 30:
        #     continue
        RA = None
        for select_chirp in select_chirps:
            sub_RA = np.load(os.path.join(RA_files, f"{frame_idx:03d}_{select_chirp:09d}.npy"))
            sub_RA = sub_RA[:, :, 0] + 1j * sub_RA[:, :, 1]
            sub_RA = np.abs(sub_RA)
            # sub_RA = np.abs(sub_RA[26:153+1, ...]) #
            # if RA is None:
            #     RA = sub_RA
            # else:
            #     RA += sub_RA

            # 保存所有Chirp-RA
            plt.close('all')
            fig = plt.figure()
            plt.imshow(sub_RA, origin='lower')
            plt.xticks([])
            plt.yticks([])
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            sub_save_path = os.path.join(save_path, f"{frame_idx:09d}_{select_chirp}.jpg")
            plt.savefig(sub_save_path, dpi=600, bbox_inches='tight', pad_inches=0)

        # 保存RD
        # RD = np.load(os.path.join(RD_files, f"{frame_idx:09d}.npy"))
        # RD = RD[:, :, 0] + 1j * RD[:, :, 1]
        # RD = np.abs(RD)
        # plt.close('all')
        # fig = plt.figure()
        # plt.imshow(RD, origin='lower')
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # sub_save_path = os.path.join(save_path, f"{frame_idx:09d}_RD.jpg")
        # plt.savefig(sub_save_path, dpi=600, bbox_inches='tight', pad_inches=0)

        # 保存AD
        # AD = np.load(os.path.join(AD_files, f"{frame_idx:09d}.npy"))
        # AD = AD[:, :, 0] + 1j * AD[:, :, 1]
        # AD = np.abs(AD)
        # plt.close('all')
        # fig = plt.figure()
        # plt.imshow(AD, origin='lower')
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        # sub_save_path = os.path.join(save_path, f"{frame_idx:09d}_AD.jpg")
        # plt.savefig(sub_save_path, dpi=600, bbox_inches='tight', pad_inches=0)

        # sub_save_path = os.path.join(save_path, f"{frame_idx:09d}.jpg")

        # visual_ra_rd_ad_from_mat(RA, RA, RA, [xs[frame_idx], ys[frame_idx]], sub_save_path)

