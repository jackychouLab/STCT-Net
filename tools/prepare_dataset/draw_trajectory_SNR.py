import os
from scipy.interpolate import make_interp_spline
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import random
import seaborn as sns
import scipy.stats as stats

# ——— Nature 期刊风格全局设置 ———
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 10,
    'axes.linewidth': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 12,
    'ytick.major.size': 12,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'grid.color': 'none',
    'legend.frameon': False,
})

def get_pkl_points(gt_path):
    with open(gt_path, 'rb') as f:
        data = pickle.load(f)[1]
        xs = []
        ys = []
        for points in data:
            x = points[0][0]
            y = points[0][1]
            xs.append(x)
            ys.append(y)
    return xs, ys

def smooth_curve(x, y):
    t = np.arange(len(x))
    spl_x = make_interp_spline(t, x, k=3)
    spl_y = make_interp_spline(t, y, k=3)
    return t, spl_x(t), spl_y(t)

def draw_all_methods_on_one_figure(methods_dict, x_g, y_g, save_path, seq_id):
    """
    methods_dict: dict like {"method_name": (x_list, y_list)}
    x_g, y_g: GT trajectory
    """
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    # Ground Truth
    t_g, x_g_smooth, y_g_smooth = smooth_curve(x_g, y_g)
    ax.plot(t_g, x_g_smooth, y_g_smooth, label="GT trajectory", color="red", linewidth=1.2)

    # All methods
    color_map = plt.cm.get_cmap('tab10', len(methods_dict))
    for idx, (method, (x_list, y_list)) in enumerate(methods_dict.items()):
        if len(x_list) < 2:
            continue  # 避免样本过少导致插值报错
        t_p, x_p_smooth, y_p_smooth = smooth_curve(x_list, y_list)
        ax.plot(t_p, x_p_smooth, y_p_smooth, label=method.upper(), color=color_map(idx), linewidth=1)

    ax.set_xlabel('Time step', labelpad=1, fontname='Times New Roman')
    ax.set_ylabel('Angle (bin)', labelpad=1, fontname='Times New Roman')
    ax.set_zlabel('Range (bin)', labelpad=0, fontname='Times New Roman')

    ax.tick_params(axis='x', direction='in', pad=0.5, size=3, width=0.5)
    ax.tick_params(axis='y', direction='in', pad=0.5, size=3, width=0.5)
    ax.tick_params(axis='z', direction='in', pad=0.5, size=3, width=0.5)

    ax.set_xlim(0, 300)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)

    ax.view_init(elev=30, azim=45)
    ax.dist = 5

    ax.legend(loc='upper left', fontsize=8, ncol=1)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{seq_id}_ALL_methods_trajectory.png"), dpi=1200)
    plt.close(fig)


def draw_trajectorys_curve(x_p, y_p, x_g, y_g, save_path, seq_id, method):
    t_p, x_p_smooth, y_p_smooth = smooth_curve(x_p, y_p)
    t_g, x_g_smooth, y_g_smooth = smooth_curve(x_g, y_g)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(t_p, x_p_smooth, y_p_smooth, label=method.upper(), color="#0072B2")
    ax.plot(t_g, x_g_smooth, y_g_smooth, label="GT trajectory", color="red")

    ax.set_xlabel('Time step', labelpad=1, fontname='Times New Roman')
    ax.set_ylabel('Angle (bin)', labelpad=1, fontname='Times New Roman')
    ax.set_zlabel('Range (bin)', labelpad=0, fontname='Times New Roman')

    ax.tick_params(axis='x', direction='in', pad=0.5, size=3, width=0.5)
    ax.tick_params(axis='y', direction='in', pad=0.5, size=3, width=0.5)
    ax.tick_params(axis='z', direction='in', pad=0.5, size=3, width=0.5)

    ax.set_xlim(0, 300)
    ax.set_ylim(0, 128)
    ax.set_zlim(0, 128)

    ax.view_init(elev=30, azim=45)
    ax.dist = 5

    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{seq_id}_{method}_trajectory.png"), dpi=1200)
    plt.close(fig)

def draw_SNR_curve(snr_list, save_path, seq_name):
    os.makedirs(save_path, exist_ok=True)
    x = np.arange(len(snr_list))
    y = np.array(snr_list)

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(x, y, color="#0072B2", linewidth=1)

    ax.set_xlabel("Frame Index", fontsize=10, labelpad=1)
    ax.set_ylabel("SNR (dB)", fontsize=10, labelpad=1)
    ax.tick_params(axis='both', direction='in', labelsize=8, width=0.5, length=3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{seq_name}.png"), dpi=600)
    plt.close(fig)

    return np.mean(y)  # 返回该序列的平均 SNR


if __name__ == '__main__':
    start_seq_id, end_seq_id = 1, 79
    base_dir = '/mnt/Data/Share/mmUAV/trajectorySNR'
    save_trajectory_dir = '/mnt/Data/Share/mmUAV/trajectorySNR/raTrajectoryMap'
    # 0: draw trajectory  1: draw SNR
    draw_type = 1

    if draw_type == 0:
        # draw trajectory
        for seq_id in tqdm(range(start_seq_id, end_seq_id + 1)):  # 1 到 79
            seq_path = os.path.join(base_dir, 'txtData', f'uav_seqs_{seq_id}', 'det_Points', 'azimuth')
            gt_path = f"/mnt/Data/Share/mmUAV/train_test_seqs/confmaps_gt/total/uav_seqs_{seq_id}_azimuth.pkl"
            gt_x_list, gt_y_list = get_pkl_points(gt_path)

            if not os.path.exists(seq_path):
                continue  # 防止路径不存在

            all_methods_traj = {}
            for method in os.listdir(seq_path):  # 遍历方法文件夹
                method_path = os.path.join(seq_path, method)
                pre_x_list = []
                pre_y_list = []
                if not os.path.isdir(method_path):
                    continue

                for i in range(len(os.listdir(method_path))): # 遍历txt文件
                    file_name = f"{i:09d}.txt"
                    file_path = os.path.join(method_path, file_name)

                    if os.path.isfile(file_path):
                        with open(file_path, 'r') as f:
                            values = f.read().strip().split()
                            if len(values) >= 2:
                                pre_x_list.append(int(values[0]))
                                pre_y_list.append(int(values[1]))
                print(len(pre_x_list), len(pre_y_list))
                # 保存到方法字典中
                all_methods_traj[method] = (pre_x_list, pre_y_list)
                draw_trajectorys_curve(pre_x_list, pre_y_list, gt_x_list, gt_y_list, save_trajectory_dir, seq_id, method)
            # 所有方法画在一张图
            draw_all_methods_on_one_figure(all_methods_traj, gt_x_list, gt_y_list, save_trajectory_dir, seq_id)
    else:
        # draw SNR
        all_snr = []
        all_avg_snr = []
        all_seq_ids = []
        save_snr_dir = '/mnt/Data/Share/mmUAV/trajectorySNR/snrDistributionMap'
        for seq_id in tqdm(range(start_seq_id, end_seq_id + 1)):  # 1 到 79
            seq_name = f'uav_seqs_{seq_id}'
            seq_snr_path = os.path.join(base_dir, 'txtData', f'uav_seqs_{seq_id}', 'SNR_Result', 'azimuth', 'snr_result_128')

            if not os.path.exists(seq_snr_path):
                continue  # 防止路径不存在

            snr_list = []
            for i in range(len(os.listdir(seq_snr_path))):  # 遍历txt文件
                file_name = f"{i:09d}.txt"
                file_path = os.path.join(seq_snr_path, file_name)

                if os.path.isfile(file_path):
                    with open(file_path, 'r') as f:
                        values = f.read().strip().split()
                        snr_list.append(float(values[0]))
                        all_snr.append(float(values[0]))

            if len(snr_list) > 0:
                avg_snr = draw_SNR_curve(snr_list, save_snr_dir, seq_name=seq_name)
                all_avg_snr.append(avg_snr)
                all_seq_ids.append(seq_name)
                print(f"Maximum SNR of uav_seq_{seq_id} is {max(snr_list)} dB.")
                print(f"Minimum SNR of uav_seq_{seq_id} is {min(snr_list)} dB.")
                print(f"Average SNR of uav_seq_{seq_id} is {avg_snr} dB.")

        # 画所有序列的平均SNR柱状图（每个柱子颜色随机）
        fig, ax = plt.subplots(figsize=(10, 4))

        # 生成随机颜色
        colors = ["#%06x" % random.randint(0, 0xFFFFFF) for _ in all_avg_snr]

        ax.bar(all_seq_ids, all_avg_snr, color=colors, width=0.5)

        ax.set_xlabel("Sequence", fontsize=10)
        ax.set_ylabel("Average SNR (dB)", fontsize=10)
        ax.tick_params(axis='x', labelrotation=90, labelsize=6)
        ax.tick_params(axis='y', labelsize=8, width=0.5, length=3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_snr_dir, 'all_seqs_average_SNR.jpg'), dpi=600)
        plt.close(fig)
        print(f"Maximum SNR is {max(all_snr)} and Minimum SNR is {min(all_snr)} of the radarUAV-24K dataset.")

        # 绘制所有 SNR 值的正态分布图
        fig, ax = plt.subplots(figsize=(8, 6))

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True)
        plt.tight_layout()

        # 使用 seaborn 绘制直方图 + KDE 曲线
        sns.histplot(all_snr, bins=50, kde=True, stat="density", color='mediumpurple', edgecolor='black', ax=ax)

        # 拟合正态分布曲线
        mu, std = stats.norm.fit(all_snr)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p, 'r--', linewidth=2, label=f'Normal Fit\nμ={mu:.2f}, σ={std:.2f}')

        # 添加图例和标签
        ax.set_title("SNR Distribution of Dataset", fontsize=20)
        ax.set_xlabel("SNR (dB)", fontsize=18)
        ax.set_ylabel("Density", fontsize=18)
        ax.legend(fontsize=18)

        plt.tight_layout()
        plt.savefig(os.path.join(save_snr_dir, 'all_snr_distribution.jpg'), dpi=1200, bbox_inches='tight')
        plt.close(fig)







