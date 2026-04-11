import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from configs.UAV_60_radar import radar_config
from scipy.interpolate import make_interp_spline, PchipInterpolator
import random



seed = 42
np.random.seed(seed)
random.seed(seed)


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


def visual_ra_rd_ad_from_mat(rd, target_range, save_path):
    plt.close()
    range_bins = radar_config['rng_grid'][radar_config['num_crop']: -radar_config['num_crop']]

    vel_bins = radar_config['vel_grid']

    max_range = 35
    range_mask = range_bins <= max_range
    range_bins = range_bins[range_mask]

    rd = rd[range_mask, :]

    R, V = np.meshgrid(range_bins, vel_bins, indexing='ij')

    # rd = np.expand_dims(np.sqrt(rd[..., 0] ** 2 + rd[..., 1] ** 2), axis=2)
    rd = np.abs(rd)
    norm_rd = rd / np.max(rd)
    # print(np.max(norm_rd))
    # norm_rd = np.where(norm_rd > 0.5, norm_rd * 0.5, norm_rd)
    # print(np.max(norm_rd))

    x = R.flatten()
    y = V.flatten()
    z = norm_rd.T.flatten()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    segments = np.zeros((len(x), 2, 3))
    segments[:, 0, 0] = y
    segments[:, 0, 1] = x
    segments[:, 0, 2] = 0
    segments[:, 1, 0] = y
    segments[:, 1, 1] = x
    segments[:, 1, 2] = z

    colors = plt.cm.jet(z)

    line_collection = Line3DCollection(segments, color=colors, alpha=0.6)
    ax.add_collection3d(line_collection)

    target_z = 0
    target_x = vel_bins
    target_y = np.full_like(vel_bins, target_range)
    # ax.plot(target_x, target_y, [target_z] * len(vel_bins), color='r', linewidth=2, label=f"UAV Range = {target_range} m")
    # ax.legend(loc="upper right")

    z_max = 1
    vertices = [
        (min(vel_bins), target_range, 0),
        (max(vel_bins), target_range, 0),
        (max(vel_bins), target_range, z_max),
        (min(vel_bins), target_range, z_max),
    ]
    # poly = Poly3DCollection([vertices], alpha=0.3, color='red')
    # ax.add_collection3d(poly)

    # proxy = Rectangle((0, 0), 1, 1, fc="red", alpha=0.3, label=f"UAV Range = {np.round(target_range, 2)} m")
    # ax.legend(handles=[proxy], loc="upper right")

    #
    ax.set_xlabel('Velocity (m/s)', size=14)
    ax.set_ylabel('Range (m)', size=14)
    ax.set_zlabel('Normalized Amplitude', size=14)

    ax.set_xlim(min(vel_bins), max(vel_bins))
    # ax.set_ylim(min(range_bins), max(range_bins))
    ax.set_ylim(0, 35)
    ax.set_zlim(0, 1)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12)


    ax.view_init(elev=30, azim=45)
    plt.savefig(save_path, dpi=600)

    plt.close()


def get_label_deg(csv_path):
    de = pd.read_csv(csv_path, delimiter="\t", header=None, skiprows=1)
    xs = []
    ys = []
    for index, row in de.iterrows():
        cx = float(row.to_dict()[0].split(', ""cy"": ')[0].split(" ")[-1])
        cy = float(row.to_dict()[0].split(', ""cy"": ')[1].split("}")[0])
        xs.append(cx)
        ys.append(cy)

    return xs, ys

def draw_arrow_next_to_trajectory(ax, theta_smooth, r_smooth, idx, offset=0.15, scale=1.2, **arrow_kwargs):
    theta0, r0 = theta_smooth[idx], r_smooth[idx]
    dtheta = theta_smooth[idx + 1] - theta_smooth[idx - 1]
    dr = r_smooth[idx + 1] - r_smooth[idx - 1]

    vec_len = np.hypot(dtheta, dr)
    if vec_len == 0:
        return

    dtheta /= vec_len
    dr /= vec_len

    n_theta = -dr
    n_r = dtheta

    theta_start = theta0 + offset * n_theta
    r_start = r0 + offset * n_r
    theta_end = theta_start + scale * dtheta
    r_end = r_start + scale * dr

    ax.annotate(
        '',
        xy=(theta_end, r_end),
        xytext=(theta_start, r_start),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.8, **arrow_kwargs),
        zorder=10
    )


def draw_arrow_next_to_trajectory(ax, theta_smooth, r_smooth, idx, offset=0.05, scale=8, **arrow_kwargs):
    # 取轨迹点和切线方向
    theta0, r0 = theta_smooth[idx], r_smooth[idx]
    dtheta = theta_smooth[idx + 1] - theta_smooth[idx - 1]
    dr = r_smooth[idx + 1] - r_smooth[idx - 1]

    vec_len = np.hypot(dtheta, dr)
    if vec_len == 0:
        return

    # 单位切线向量
    dtheta /= vec_len
    dr /= vec_len

    # 计算法线方向（垂直于切线）
    n_theta = -dr
    n_r = dtheta

    # 箭头起点（偏移轨迹法线方向）
    theta_start = theta0 + n_theta * offset
    r_start = r0 + n_r * offset

    # 箭头终点（沿切线方向延长）
    theta_end = theta_start + dtheta * offset
    r_end = r_start + dr * scale

    ax.annotate(
        '',
        xy=(theta_end, r_end),
        xytext=(theta_start, r_start),
        arrowprops=dict(arrowstyle='->', color='red', lw=1.8, **arrow_kwargs),
        zorder=10
    )


def visual_label_tra(xs, ys, save_path):
    plt.close('all')
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'legend.fontsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'axes.linewidth': 1.,
    })

    # step1 生成高分辨率的参数化曲线
    xs = np.array(xs)
    ys = np.clip(np.array(ys), 0, 90)
    theta = xs + np.pi / 2
    x_cart = ys * np.cos(theta)
    y_cart = ys * np.sin(theta)
    t = np.linspace(0, 1, len(xs))
    t_smooth = np.linspace(0, 1, 10 * len(xs))
    spline_x = PchipInterpolator(t, x_cart)
    spline_y = PchipInterpolator(t, y_cart)
    x_smooth = spline_x(t_smooth)
    y_smooth = spline_y(t_smooth)

    # step2 计算曲线的累积弧长
    segment_lengths = np.hypot(np.diff(x_smooth), np.diff(y_smooth))
    cumulative_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    total_length = cumulative_length[-1]
    target_distances = np.arange(0, total_length, 0.5)
    t_equidistant = np.interp(target_distances, cumulative_length, t_smooth)
    x_eq = spline_x(t_equidistant)
    y_eq = spline_y(t_equidistant)
    r_eq = np.hypot(x_eq, y_eq)
    theta_eq = np.arctan2(y_eq, x_eq)
    num_markers = len(r_eq)
    alphas = np.linspace(0.1, 1.0, num=num_markers) if num_markers > 0 else [1.0]
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.scatter(theta_eq, r_eq, s=15, c='#FF0000', marker='o', zorder=3, alpha=alphas, label='UAV Position')

    # Convert back to polar coordinates
    ys_smooth = np.sqrt(x_smooth ** 2 + y_smooth ** 2)
    theta_smooth = np.unwrap(np.arctan2(y_smooth, x_smooth))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta_smooth, ys_smooth, color=(1.0, 0.0, 0.0, 1.0), linewidth=1, linestyle='--', zorder=2, label='UAV Trajectory')

    # Set radar range and limits
    rand_max_range = np.random.randint(5, 16)
    max_range = min(90, np.ceil(np.max(ys_smooth) / rand_max_range) * rand_max_range) if np.max(ys_smooth) > 0 else 10
    ax.set_rlim(0, max_range)
    ax.set_thetalim(np.radians(30), np.radians(150))
    ax.set_xticks(np.radians(np.arange(30, 151, 15)))
    ax.set_xticklabels([f'{int(-x)}°' for x in np.arange(-60, 61, 15)])
    ax.set_rlabel_position(90)
    ax.set_yticks(np.arange(0, max_range + 1, 30))
    ax.set_yticks([])
    ax.grid(color=(0.576, 0.439, 0.858, 1.0), linestyle=':', linewidth=0.5, zorder=2)
    # ax.spines['polar'].set_visible(False)
    ax.set_facecolor((0.529, 0.808, 0.922, 0.8))

    # Add title and legend
    # plt.title('Drone Trajectory on Radar (Max Range: 90m)', pad=20)
    # ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Adjust layout and save
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1200, bbox_inches='tight', pad_inches=0)
    plt.close()

# data_root = "/home/jackychou/dataset/UAVRadar"
# idx_list = [i for i in range(1, 1 + 79)]
# use_filter = 11
# frames_list = [i for i in range(0, 300)]
# idx_list = [10, 11, 32, 50]

# for seq_idx in idx_list:
#     # sub_path = os.path.join(data_root, f"uav_seqs_{str(seq_idx)}", f"python_slice_frame_{str(use_filter)}", "azimuth")

#     save_path = f"/mnt/d/paperv1/visual_3D_labels_tra/{str(seq_idx)}.jpg"
#     visual_label_tra(xs, ys, save_path)

root = '/home/jackychou/dataset/UAVRadar/uav_seqs_9'
csv_path = os.path.join(root, "annot", "rodnet_labels_128_rad.csv")
xs, ys = get_label_deg(csv_path)
for idx in range(30, 31):
    rd_path = os.path.join(root, 'python_slice_frame_11/azimuth/raw_frame_RD', f'{idx:09d}.npy')
    rd = np.load(rd_path)
    save_path = f"/mnt/d/paperv1/visual_3D_labels_tra/{idx+1:09d}.jpg"
    visual_ra_rd_ad_from_mat(rd, ys[idx], save_path)


