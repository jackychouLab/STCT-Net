import os
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat



def check_path(x):
    if type(x) is str:
        x = [x]
    assert x is list
    for xi in x:
        if os.path.exists(xi) is False:
            os.mkdir(xi)


def read_mat(x):
    mat_data = loadmat(x)
    if 'RA_data' in mat_data:
        return np.array(mat_data['RA_data'])
    if 'RD_data' in mat_data:
        return np.array(mat_data['RD_data'])
    if 'AD_data' in mat_data:
        return np.array(mat_data['AD_data'])

    return None


def get_csv_points(x):
    df = pd.read_csv(x, delimiter='\t', header=None, skiprows=1)
    xs = []
    ys = []
    for index, row in df.iterrows():
        cx = int(row.to_dict()[0].split(', ""cy"": ')[0].split(" ")[-1])
        cy = int(row.to_dict()[0].split(', ""cy"": ')[1].split("}")[0])
        cx = cx * 2
        cy = 128 - cy + 5
        xs.append(cx)
        ys.append(cy)

    return xs, ys


def get_txt_points(x):
    pred_label = {}
    with open(x, "r") as f:
        for num, line in enumerate(f):
            idx, a, r = line.split('\t')
            idx = int(idx)
            a = int(a)
            r = int(r)
            if idx not in pred_label:
                pred_label[idx] = []
                pred_label[idx].append([a, r])
            else:
                pred_label[idx].append([a, r])

    return pred_label


def read_npy(x):
    npy_data = np.load(x)

    return npy_data


def visual_ra(ra):
    plt.close()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ra = np.sqrt(ra[:, :, 0] ** 2 + ra[:, :, 1] ** 2)
    plt.imshow(ra, origin='lower')
    ax1.set_title("RA")
    plt.tight_layout()
    plt.show()
    plt.close()


def visual_ra_w_points(ra, points):
    plt.close()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ra = np.sqrt(ra[:, :, 0] ** 2 + ra[:, :, 1] ** 2)
    plt.imshow(ra, origin='lower')
    for point_idx in range(len(points)):
        if point_idx == 0:
            color_type = 'red'
        else:
            color_type = 'blue'
        circle = plt.Circle((points[point_idx][0], points[point_idx][1]), 2, color=color_type, fill=True)
        ax1.add_artist(circle)
    ax1.set_title("RA")
    plt.tight_layout()
    plt.show()
    plt.close()


def cal_ols(object_h, object_l, cls_size, radar_config):
    range_grid = radar_config['rng_grid']
    angle_grid = radar_config['agl_grid']
    rid1 = object_h[1]
    aid1 = object_h[0]
    rid2 = object_l[1]
    aid2 = object_l[0]
    x1 = rid1 * np.sin(aid1)
    y1 = rid1 * np.cos(aid1)
    x2 = rid2 * np.sin(aid2)
    y2 = rid2 * np.cos(aid2)
    dx = x1 - x2
    dy = y1 - y2
    s_square = x1 ** 2 + y1 ** 2
    kappa = cls_size / 100
    e = (dx ** 2 + dy ** 2) / 2 / (s_square * kappa)
    ols = np.exp(-e)

    return ols


def getadcDataFromDCA1000(fileName, radar_config):
    adcDataName = os.listdir(fileName)
    adcDataName.sort(key=lambda adcDataName: adcDataName.split('_')[-1][:-4])
    adc_Data = None
    for num in range(len(adcDataName)):
        if adc_Data is None:
            adc_Data = np.fromfile(os.path.join(fileName, adcDataName[num]), dtype=np.int16)
        else:
            adc_Data = np.concatenate((adc_Data, np.fromfile(os.path.join(fileName, adcDataName[num]), dtype=np.int16)))

    assert adc_Data.shape[0] == radar_config['Tx'] * radar_config['Rx'] * radar_config['duration'] * radar_config['Lanes'] * radar_config['frame_number'] * radar_config['samples'] * radar_config['loop']

    adc_Data = np.transpose(adc_Data.reshape(-1, 2, 2), axes=(0, 2, 1)).reshape(-1, 2)
    adc_Data = adc_Data[:, 0] + (np.sqrt(-1 + 0j) * adc_Data[:, 1])

    return adc_Data


def generateHeatmap_RA(frame, radar_config):
    R_win = np.hamming(frame.shape[0])
    R_FFT = np.fft.fft(frame * R_win[:, np.newaxis, np.newaxis], radar_config['fft_Range'], axis=0)
    RA_slice = produce_RA_slice(R_FFT, radar_config)

    return RA_slice[radar_config['num_crop']:-radar_config['num_crop'], ...]


def produce_RA_slice(x, radar_config):
    hamming_win = np.hamming(x.shape[1])
    RA_raw = np.fft.fft(x * hamming_win[np.newaxis, :, np.newaxis], radar_config['fft_Angle'], axis=1)
    RA_raw = np.fft.fftshift(RA_raw, axes=1)
    RA_real = np.expand_dims(RA_raw.real, axis=3)
    RA_imag = np.expand_dims(RA_raw.imag, axis=3)
    RA_slice = np.float32(np.concatenate((RA_real, RA_imag), axis=3))

    return RA_slice


def visual_person_RA(ra, save_name=None):
    plt.close()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0, 0])
    ra = np.sqrt(ra[:, :, :, 0] ** 2 + ra[:, :, :, 1] ** 2)
    ra = ra[:, :, ra.shape[-1] // 2]
    plt.imshow(ra, origin='lower')
    ax1.set_title("RA")
    plt.tight_layout()
    if save_name is not None:
        plt.savefig(save_name, dpi=600)
    else:
        plt.show()


def visual_ra_rd_ad_from_mat(ra, rd, ad, save_path=None):
    plt.close()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    show_ra = np.sqrt(ra[..., 0] ** 2 + ra[..., 1] ** 2)
    plt.imshow(show_ra, origin='lower')
    ax1.set_title("RA")

    ax2 = fig.add_subplot(gs[0, 1])
    show_rd = np.sqrt(rd[..., 0] ** 2 + rd[..., 1] ** 2)
    plt.imshow(show_rd, origin='lower')
    ax2.set_title('RD')

    ax3 = fig.add_subplot(gs[0, 2])
    show_ad = np.sqrt(ad[..., 0] ** 2 + ad[..., 1] ** 2)
    plt.imshow(show_ad, origin='lower')
    ax3.set_title('AD')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    else:
        plt.show()
    plt.close()


def visual_ra_rd_ad_gt_from_mat(ra, rd, ad, gt_label, save_path=None):
    plt.close()
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    show_ra = np.sqrt(ra[:, :, 0] ** 2 + ra[:, :, 1] ** 2)
    plt.imshow(show_ra, origin='lower')
    ax1.set_title("RA")
    plt.plot(gt_label[1], gt_label[0], marker='*', color='red', markersize=5)

    ax2 = fig.add_subplot(gs[0, 1])
    show_rd = np.sqrt(rd[:, :, 0] ** 2 + rd[:, :, 1] ** 2)
    plt.imshow(show_rd, origin='lower')
    ax2.set_title('RD')

    ax3 = fig.add_subplot(gs[0, 2])
    show_ad = np.sqrt(ad[:, :, 0] ** 2 + ad[:, :, 1] ** 2)
    plt.imshow(show_ad, origin='lower')
    ax3.set_title('AD')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=600)
    else:
        plt.show()
    plt.close()