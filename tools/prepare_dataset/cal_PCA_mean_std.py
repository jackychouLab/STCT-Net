import os
import numpy as np
from tools.instruments.instruments import read_npy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor



root = "/home/jackychou/Zhou/dataset/UAVRadar"
trainIdx = [1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 66, 67, 68, 69, 70, 72, 75, 76, 78, 79]
trainDataType = 'python'
use_cluster_filters = [11]
radarType = 'azimuth'

MAX_WORKERS = 8
THRESHOLDS = [1, 4, 8, 16, 32, 64, 128]
LABELS = {1: "PCA1", 4: "PCA4", 8: "PCA8", 16: "PCA16", 32: "PCA32", 64: "PCA64", 128: "PCA128"}


def build_tasks(root, trainIdx, use_cluster_filter, radarType):
    tasks = []
    for idx in trainIdx:
        seqPath = os.path.join(
            root,
            f'uav_seqs_{idx}',
            f'{trainDataType}_slice_frame_{use_cluster_filter}_PCAv1',
            radarType
        )
        raPath = os.path.join(seqPath, 'raw_frame_RA')

        for frame_idx in range(0, 300):
            for chirp_idx in range(1, 129):
                sub_raPath = os.path.join(raPath, f'{frame_idx:03d}_{chirp_idx:09d}.npy')
                tasks.append((sub_raPath, chirp_idx))
    return tasks


def process_file_mean(task):
    file_path, chirp_idx = task
    data = read_npy(file_path)

    data_real = data[..., 0]
    data_imag = data[..., 1]
    data_complex = data_real + 1j * data_imag
    data_magnitude = np.abs(data_complex)
    data_phase = np.arctan2(data_imag, data_real)

    sum_real = np.sum(data_real, dtype=np.float128)
    sum_imag = np.sum(data_imag, dtype=np.float128)
    sum_mag = np.sum(data_magnitude, dtype=np.float128)
    sum_phase = np.sum(data_phase, dtype=np.float128)
    count = np.float128(data_real.size)
    sum_complex = np.sum(data_complex, dtype=np.complex256)

    contribs = []
    for th in THRESHOLDS:
        if chirp_idx <= th:
            contribs.append((sum_real, sum_imag, sum_mag, sum_phase, count, sum_complex))
        else:
            contribs.append(None)
    return contribs


def process_file_std(task):
    file_path, chirp_idx, means_pack = task
    data = read_npy(file_path)

    data_real = data[..., 0]
    data_imag = data[..., 1]
    data_complex = data_real + 1j * data_imag
    data_magnitude = np.abs(data_complex)
    data_phase = np.arctan2(data_imag, data_real)


    flat_real = data_real.reshape(-1)
    flat_imag = data_imag.reshape(-1)
    flat_mag = data_magnitude.reshape(-1)
    flat_phase = data_phase.reshape(-1)
    count = np.float128(flat_real.size)
    flat_complex = data_complex.reshape(-1)

    contribs = []
    for th in THRESHOLDS:
        if chirp_idx <= th:
            mean_real, mean_imag, mean_mag, mean_phase, mean_complex = means_pack[th]

            diff_real = flat_real - mean_real
            diff_imag = flat_imag - mean_imag
            diff_mag = flat_mag - mean_mag
            diff_phase = flat_phase - mean_phase
            diff_complex = flat_complex - mean_complex

            ss_real = np.sum(diff_real ** 2, dtype=np.float128)
            ss_imag = np.sum(diff_imag ** 2, dtype=np.float128)
            ss_mag = np.sum(diff_mag ** 2, dtype=np.float128)
            ss_phase = np.sum(diff_phase ** 2, dtype=np.float128)
            ss_complex = np.sum(np.abs(diff_complex) ** 2, dtype=np.float128)

            contribs.append((ss_real, ss_imag, ss_mag, ss_phase, count, ss_complex))
        else:
            contribs.append(None)
    return contribs


for use_cluster_filter in use_cluster_filters:
    raSum = {
        th: [np.float128(0.), np.float128(0.), np.float128(0.), np.float128(0.), np.float128(0.), np.complex256(0.)]
        for th in THRESHOLDS
    }

    tasks = build_tasks(root, trainIdx, use_cluster_filter, radarType)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for contribs in tqdm(
            executor.map(process_file_mean, tasks),
            total=len(tasks),
            desc=f"[cluster {use_cluster_filter}] mean"
        ):
            for th, contrib in zip(THRESHOLDS, contribs):
                if contrib is None:
                    continue
                sr, si, smag, sph, cnt, sc = contrib
                acc = raSum[th]
                acc[0] += sr
                acc[1] += si
                acc[2] += smag
                acc[3] += sph
                acc[4] += cnt
                acc[5] += sc

    means = {}
    for th in THRESHOLDS:
        sum_real, sum_imag, sum_mag, sum_phase, cnt, sum_complex = raSum[th]
        if cnt > 0:
            mu_real = sum_real / cnt
            mu_imag = sum_imag / cnt
            mu_mag = sum_mag / cnt
            mu_phase = sum_phase / cnt
            mu_complex = sum_complex / cnt
            means[th] = {
                "real": np.float32(mu_real),
                "imag": np.float32(mu_imag),
                "mag":  np.float32(mu_mag),
                "phase": np.float32(mu_phase),
                "complex": np.complex64(mu_complex),
                "count": np.float128(cnt),
            }
        else:
            means[th] = {
                "real": np.float32(0.),
                "imag": np.float32(0.),
                "mag":  np.float32(0.),
                "phase": np.float32(0.),
                "complex": np.complex64(0.),
                "count": np.float128(0.),
            }

    means_pack = {
        th: (means[th]["real"], means[th]["imag"], means[th]["mag"], means[th]["phase"], means[th]["complex"])
        for th in THRESHOLDS
    }

    raVarSum = {
        th: [np.float128(0.), np.float128(0.), np.float128(0.), np.float128(0.), np.float128(0.), np.float128(0.)]
        for th in THRESHOLDS
    }

    tasks_std = [(path, chirp_idx, means_pack) for (path, chirp_idx) in tasks]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for contribs in tqdm(
            executor.map(process_file_std, tasks_std),
            total=len(tasks_std),
            desc=f"[cluster {use_cluster_filter}] std"
        ):
            for th, contrib in zip(THRESHOLDS, contribs):
                if contrib is None:
                    continue
                ss_real, ss_imag, ss_mag, ss_phase, cnt, ss_complex = contrib
                acc = raVarSum[th]
                acc[0] += ss_real
                acc[1] += ss_imag
                acc[2] += ss_mag
                acc[3] += ss_phase
                acc[4] += cnt
                acc[5] += ss_complex

    # 用平方和 / count 得到方差，再开方得到标准差
    # stds[th] = {'real': σ_real, 'imag': σ_imag, 'mag': σ_mag, 'phase': σ_phase, 'complex': σ_complex}
    stds = {}
    for th in THRESHOLDS:
        ss_real, ss_imag, ss_mag, ss_phase, cnt, ss_complex = raVarSum[th]
        if cnt > 0:
            var_real = ss_real / cnt
            var_imag = ss_imag / cnt
            var_mag = ss_mag / cnt
            var_phase = ss_phase / cnt
            var_complex = ss_complex / cnt

            std_real = np.float32(np.sqrt(var_real))
            std_imag = np.float32(np.sqrt(var_imag))
            std_mag = np.float32(np.sqrt(var_mag))
            std_phase = np.float32(np.sqrt(var_phase))
            std_complex = np.float32(np.sqrt(var_complex))

            stds[th] = {
                "real": std_real,
                "imag": std_imag,
                "mag": std_mag,
                "phase": std_phase,
                "complex": std_complex,
            }
        else:
            stds[th] = {
                "real": np.float128(0.),
                "imag": np.float128(0.),
                "mag": np.float128(0.),
                "phase": np.float128(0.),
                "complex": np.float128(0.),
            }

    # ----------- 5. 写文件 -----------
    out_path = "/home/jackychou/Zhou/UAVRadar_mean_std/PCAv1.txt"
    if not os.path.exists(out_path):
        with open(out_path, "w") as f:
            pass

    with open(out_path, "a") as f:
        f.write("##################################################################### \n")
        for th in THRESHOLDS:
            m = means[th]
            s = stds[th]
            label = LABELS[th]

            f.write(f"{use_cluster_filter}_{label} \n")
            f.write(f"RA:\n实部均值={float(m['real'])}\n实部标准差={float(s['real'])}\n")
            f.write(f"RA:\n虚部均值={float(m['imag'])}\n虚部标准差={float(s['imag'])}\n")
            f.write(
                f"RA:\n复数实部均值={float(m['complex'].real)}\n"
                f"复数虚部均值={float(m['complex'].imag)}\n"
                f"标准差={float(s['complex'])}\n"
            )
        f.write("##################################################################### \n")
