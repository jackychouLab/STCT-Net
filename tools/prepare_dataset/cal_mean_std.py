import os
import numpy as np
from tools.instruments.instruments import read_npy, read_mat
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
import time



def _collect_dir_files(dir_path, exts):
    out = []
    for fn in sorted(os.listdir(dir_path)):
        p = os.path.join(dir_path, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in exts:
            out.append(p)
    return out


def collect_files(root, trainIdx, use_cluster_filter, radarType):
    ra_files, rd_files, ad_files = [], [], []

    for idx in tqdm(trainIdx, desc=f"Collect files (cluster {use_cluster_filter})", dynamic_ncols=True):
        seqPath = os.path.join(root, f'uav_seqs_{idx}', f'{trainDataType}_slice_frame_{use_cluster_filter}', radarType)
        raPath = os.path.join(seqPath, 'raw_frame_RA')
        rdPath = os.path.join(seqPath, 'raw_frame_RD')
        adPath = os.path.join(seqPath, 'raw_frame_AD')

        if os.path.isdir(raPath):
            ra_files.extend(_collect_dir_files(raPath, exts={".npy", ".mat"}))

        if os.path.isdir(rdPath):
            rd_files.extend(_collect_dir_files(rdPath, exts={".npy", ".mat"}))

        if os.path.isdir(adPath):
            ad_files.extend(_collect_dir_files(adPath, exts={".npy", ".mat"}))

    return ra_files, rd_files, ad_files


def load_data(file_path, trainDataType):
    if trainDataType == "matlab":
        data = read_mat(file_path)
    else:
        data = read_npy(file_path)
    return data


def process_file_mean(file_path, trainDataType):
    data = load_data(file_path, trainDataType)

    data_real = data[..., 0]
    data_imag = data[..., 1]
    data_complex = data_real + 1j * data_imag
    data_magnitude = np.hypot(data_real, data_imag)
    data_phase = np.arctan2(data_imag, data_real)

    sum_real = np.sum(data_real, dtype=np.float128)
    sum_imag = np.sum(data_imag, dtype=np.float128)
    sum_mag = np.sum(data_magnitude, dtype=np.float128)
    sum_phase = np.sum(data_phase, dtype=np.float128)
    count = np.float128(data_real.size)
    sum_complex = np.sum(data_complex, dtype=np.complex256)

    return sum_real, sum_imag, sum_mag, sum_phase, count, sum_complex


def process_file_std(file_path, trainDataType, mean_real, mean_imag, mean_mag, mean_phase, mean_complex):
    data = load_data(file_path, trainDataType)

    data_real = data[..., 0]
    data_imag = data[..., 1]
    data_complex = data_real + 1j * data_imag
    data_magnitude = np.hypot(data_real, data_imag)
    data_phase = np.arctan2(data_imag, data_real)

    flat_real = data_real.reshape(-1)
    flat_imag = data_imag.reshape(-1)
    flat_complex = data_complex.reshape(-1)
    flat_mag = data_magnitude.reshape(-1)
    flat_phase = data_phase.reshape(-1)

    mr = np.float64(mean_real)
    mi = np.float64(mean_imag)
    mm = np.float64(mean_mag)
    mp = np.float64(mean_phase)
    mc = np.complex128(mean_complex)

    dr = flat_real.astype(np.float64) - mr
    di = flat_imag.astype(np.float64) - mi
    dm = flat_mag.astype(np.float64) - mm
    dp = flat_phase.astype(np.float64) - mp

    sum_sq_real = np.sum(dr * dr, dtype=np.float128)
    sum_sq_imag = np.sum(di * di, dtype=np.float128)
    sum_sq_mag = np.sum(dm * dm, dtype=np.float128)
    sum_sq_phase = np.sum(dp * dp, dtype=np.float128)

    dc = flat_complex.astype(np.complex128) - mc
    sum_sq_complex = np.sum((dc.real * dc.real + dc.imag * dc.imag).astype(np.float64), dtype=np.float128)

    count = np.float128(flat_real.size)

    return sum_sq_real, sum_sq_imag, sum_sq_mag, sum_sq_phase, count, sum_sq_complex


root = "/home/jackychou/dataset/UAVRadar"
trainIdx = [1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 66, 67, 68, 69, 70, 72, 75, 76, 78, 79]
trainDataType = 'python'  # 'matlab' 'python' '4dfft' 'PCA'
use_cluster_filters = [11]
radarType = 'azimuth'  # 'azmimuth' 'elevation'
MAX_WORKERS = 16
for use_cluster_filter in use_cluster_filters:
    ra_files, rd_files, ad_files = collect_files(
        root, trainIdx, use_cluster_filter, radarType
    )

    raMean = [np.float128(0.) for _ in range(4)] + [np.float128(0.)] + [np.complex256(0.)]
    rdMean = [np.float128(0.) for _ in range(4)] + [np.float128(0.)] + [np.complex256(0.)]
    adMean = [np.float128(0.) for _ in range(4)] + [np.float128(0.)] + [np.complex256(0.)]

    if len(ra_files) > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            iterator = executor.map(
                process_file_mean,
                ra_files,
                repeat(trainDataType),
            )
            for sum_real, sum_imag, sum_mag, sum_phase, count, sum_complex in tqdm(
                    iterator,
                    total=len(ra_files),
                    desc=f"RA mean (cluster {use_cluster_filter})",
                    dynamic_ncols=True
            ):
                raMean[0] += sum_real
                raMean[1] += sum_imag
                raMean[2] += sum_mag
                raMean[3] += sum_phase
                raMean[4] += count
                raMean[5] += sum_complex

    if len(rd_files) > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            iterator = executor.map(
                process_file_mean,
                rd_files,
                repeat(trainDataType),
            )
            for sum_real, sum_imag, sum_mag, sum_phase, count, sum_complex in tqdm(
                    iterator,
                    total=len(rd_files),
                    desc=f"RD mean (cluster {use_cluster_filter})",
                    dynamic_ncols=True
            ):
                rdMean[0] += sum_real
                rdMean[1] += sum_imag
                rdMean[2] += sum_mag
                rdMean[3] += sum_phase
                rdMean[4] += count
                rdMean[5] += sum_complex

    if len(ad_files) > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            iterator = executor.map(
                process_file_mean,
                ad_files,
                repeat(trainDataType),
            )
            for sum_real, sum_imag, sum_mag, sum_phase, count, sum_complex in tqdm(
                    iterator,
                    total=len(ad_files),
                    desc=f"AD mean (cluster {use_cluster_filter})",
                    dynamic_ncols=True
            ):
                adMean[0] += sum_real
                adMean[1] += sum_imag
                adMean[2] += sum_mag
                adMean[3] += sum_phase
                adMean[4] += count
                adMean[5] += sum_complex

    if raMean[4] > 0:
        cnt = raMean[4]
        raMean[0] = np.float64(raMean[0] / cnt)
        raMean[1] = np.float64(raMean[1] / cnt)
        raMean[2] = np.float64(raMean[2] / cnt)
        raMean[3] = np.float64(raMean[3] / cnt)
        raMean[5] = np.complex128(raMean[5] / cnt)

    if rdMean[4] > 0:
        cnt = rdMean[4]
        rdMean[0] = np.float64(rdMean[0] / cnt)
        rdMean[1] = np.float64(rdMean[1] / cnt)
        rdMean[2] = np.float64(rdMean[2] / cnt)
        rdMean[3] = np.float64(rdMean[3] / cnt)
        rdMean[5] = np.complex128(rdMean[5] / cnt)

    if adMean[4] > 0:
        cnt = adMean[4]
        adMean[0] = np.float64(adMean[0] / cnt)
        adMean[1] = np.float64(adMean[1] / cnt)
        adMean[2] = np.float64(adMean[2] / cnt)
        adMean[3] = np.float64(adMean[3] / cnt)
        adMean[5] = np.complex128(adMean[5] / cnt)

    raStd = [np.float128(0.) for _ in range(4)] + [np.float128(0.)] + [np.float128(0.)]
    rdStd = [np.float128(0.) for _ in range(4)] + [np.float128(0.)] + [np.float128(0.)]
    adStd = [np.float128(0.) for _ in range(4)] + [np.float128(0.)] + [np.float128(0.)]

    if len(ra_files) > 0 and raMean[4] > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            iterator = executor.map(
                process_file_std,
                ra_files,
                repeat(trainDataType),
                repeat(raMean[0]),
                repeat(raMean[1]),
                repeat(raMean[2]),
                repeat(raMean[3]),
                repeat(raMean[5]),
            )
            for ss_real, ss_imag, ss_mag, ss_phase, count, ss_complex in tqdm(
                    iterator,
                    total=len(ra_files),
                    desc=f"RA std (cluster {use_cluster_filter})",
                    dynamic_ncols=True
            ):
                raStd[0] += ss_real
                raStd[1] += ss_imag
                raStd[2] += ss_mag
                raStd[3] += ss_phase
                raStd[4] += count
                raStd[5] += ss_complex

    if len(rd_files) > 0 and rdMean[4] > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            iterator = executor.map(
                process_file_std,
                rd_files,
                repeat(trainDataType),
                repeat(rdMean[0]),
                repeat(rdMean[1]),
                repeat(rdMean[2]),
                repeat(rdMean[3]),
                repeat(rdMean[5]),
            )
            for ss_real, ss_imag, ss_mag, ss_phase, count, ss_complex in tqdm(
                    iterator,
                    total=len(rd_files),
                    desc=f"RD std (cluster {use_cluster_filter})",
                    dynamic_ncols=True
            ):
                rdStd[0] += ss_real
                rdStd[1] += ss_imag
                rdStd[2] += ss_mag
                rdStd[3] += ss_phase
                rdStd[4] += count
                rdStd[5] += ss_complex

    if len(ad_files) > 0 and adMean[4] > 0:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            iterator = executor.map(
                process_file_std,
                ad_files,
                repeat(trainDataType),
                repeat(adMean[0]),
                repeat(adMean[1]),
                repeat(adMean[2]),
                repeat(adMean[3]),
                repeat(adMean[5]),
            )
            for ss_real, ss_imag, ss_mag, ss_phase, count, ss_complex in tqdm(
                    iterator,
                    total=len(ad_files),
                    desc=f"AD std (cluster {use_cluster_filter})",
                    dynamic_ncols=True
            ):
                adStd[0] += ss_real
                adStd[1] += ss_imag
                adStd[2] += ss_mag
                adStd[3] += ss_phase
                adStd[4] += count
                adStd[5] += ss_complex

    if raStd[4] > 0:
        cnt = raStd[4]
        raStd[0] = np.float32(np.sqrt(raStd[0] / cnt))
        raStd[1] = np.float32(np.sqrt(raStd[1] / cnt))
        raStd[2] = np.float32(np.sqrt(raStd[2] / cnt))
        raStd[3] = np.float32(np.sqrt(raStd[3] / cnt))
        raStd[5] = np.float32(np.sqrt(raStd[5] / cnt))

    if rdStd[4] > 0:
        cnt = rdStd[4]
        rdStd[0] = np.float32(np.sqrt(rdStd[0] / cnt))
        rdStd[1] = np.float32(np.sqrt(rdStd[1] / cnt))
        rdStd[2] = np.float32(np.sqrt(rdStd[2] / cnt))
        rdStd[3] = np.float32(np.sqrt(rdStd[3] / cnt))
        rdStd[5] = np.float32(np.sqrt(rdStd[5] / cnt))

    if adStd[4] > 0:
        cnt = adStd[4]
        adStd[0] = np.float32(np.sqrt(adStd[0] / cnt))
        adStd[1] = np.float32(np.sqrt(adStd[1] / cnt))
        adStd[2] = np.float32(np.sqrt(adStd[2] / cnt))
        adStd[3] = np.float32(np.sqrt(adStd[3] / cnt))
        adStd[5] = np.float32(np.sqrt(adStd[5] / cnt))

    out_path = "/home/jackychou/Zhou/dataset/UAVRadar_mean_std/Uniform.txt"

    if os.path.exists(out_path) is False:
        with open(out_path, "w") as f:
            pass

    with open(out_path, "a") as f:
        f.write(f"{use_cluster_filter}\n")
        f.write(f"{trainDataType}_{use_cluster_filter}\n")

        f.write(f"RA:\n实部均值={np.float32(raMean[0])}\n实部标准差={np.float32(raStd[0])}\n")
        f.write(f"RA:\n虚部均值={np.float32(raMean[1])}\n虚部标准差={np.float32(raStd[1])}\n")
        f.write(f"RA:\n模均值={np.float32(raMean[2])}\n模标准差={np.float32(raStd[2])}\n")
        f.write(f"RA:\n相位均值={np.float32(raMean[3])}\n相位标准差={np.float32(raStd[3])}\n")
        f.write(f"RA:\n复数实部均值={np.float32(raMean[5].real)}\n复数虚部均值={np.float32(raMean[5].imag)}\n标准差={np.float32(raStd[5])}\n")

        f.write(f"RD:\n实部均值={np.float32(rdMean[0])}\n实部标准差={np.float32(rdStd[0])}\n")
        f.write(f"RD:\n虚部均值={np.float32(rdMean[1])}\n虚部标准差={np.float32(rdStd[1])}\n")
        f.write(f"RD:\n模均值={np.float32(rdMean[2])}\n模标准差={np.float32(rdStd[2])}\n")
        f.write(f"RD:\n相位均值={np.float32(rdMean[3])}\n相位标准差={np.float32(rdStd[3])}\n")
        f.write(f"RD:\n复数实部均值={np.float32(rdMean[5].real)}\n复数虚部均值={np.float32(rdMean[5].imag)}\n标准差={np.float32(rdStd[5])}\n")

        f.write(f"AD:\n实部均值={np.float32(adMean[0])}\n实部标准差={np.float32(adStd[0])}\n")
        f.write(f"AD:\n虚部均值={np.float32(adMean[1])}\n虚部标准差={np.float32(adStd[1])}\n")
        f.write(f"AD:\n模均值={np.float32(adMean[2])}\n模标准差={np.float32(adStd[2])}\n")
        f.write(f"AD:\n相位均值={np.float32(adMean[3])}\n相位标准差={np.float32(adStd[3])}\n")
        f.write(f"AD:\n复数实部均值={np.float32(adMean[5].real)}\n复数虚部均值={np.float32(adMean[5].imag)}\n标准差={np.float32(adStd[5])}\n")
