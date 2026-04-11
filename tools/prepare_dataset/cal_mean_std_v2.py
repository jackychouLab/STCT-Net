import os
import numpy as np
from tools.instruments.instruments import read_npy, read_mat
from tqdm import tqdm

root = "/home/jackychou/Zhou/dataset/UAV-Radar"
trainIdx = [1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 66, 67, 68, 69, 70, 72, 75, 76, 78, 79]
trainDataType = 'python'  # 'matlab' 'python' '4dfft' 'PCA'
use_cluster_filters = [11]
radarType = 'azimuth'     # 'azimuth' 'elevation'

only_RD_AD = False

def list_files(folder):
    if not os.path.isdir(folder):
        return []
    # 只取常见扩展；也可不筛选
    return sorted([f for f in os.listdir(folder) if not f.startswith('.')])

def load_pair(filePath, loader):
    if loader == "matlab":
        data = read_mat(filePath)
    else:
        data = read_npy(filePath)
    # 统一 float64 计算精度；最后统计再缩放回去
    data = data.astype(np.float64, copy=False)
    # data[...,0] 实部，data[...,1] 虚部
    real = data[..., 0]
    imag = data[..., 1]
    comp = real + 1j * imag
    mag  = np.abs(comp)
    pha  = np.arctan2(imag, real)  # 线性相位（注意：非环状均值）
    return real, imag, mag, pha, comp

for use_cluster_filter in use_cluster_filters:

    # ---------- Pass 0: 找全局最大幅值，确定缩放因子（按 2 的幂） ----------
    ra_max_amp = 0.0
    rd_max_amp = 0.0
    ad_max_amp = 0.0

    for idx in tqdm(trainIdx, desc=f"[Pass0] scan max amp {use_cluster_filter}"):
        seqPath = os.path.join(root, f'uav_seqs_{idx}', f'{trainDataType}_slice_frame_{use_cluster_filter}', radarType)
        raPath = os.path.join(seqPath, 'raw_frame_RA')
        rdPath = os.path.join(seqPath, 'raw_frame_RD')
        adPath = os.path.join(seqPath, 'raw_frame_AD')

        if not only_RD_AD:
            for fn in list_files(raPath):
                real, imag, mag, pha, comp = load_pair(os.path.join(raPath, fn), trainDataType)
                m = np.max(mag) if mag.size else 0.0
                if m > ra_max_amp: ra_max_amp = m

        for fn in list_files(rdPath):
            real, imag, mag, pha, comp = load_pair(os.path.join(rdPath, fn), trainDataType)
            m = np.max(mag) if mag.size else 0.0
            if m > rd_max_amp: rd_max_amp = m

        for fn in list_files(adPath):
            real, imag, mag, pha, comp = load_pair(os.path.join(adPath, fn), trainDataType)
            m = np.max(mag) if mag.size else 0.0
            if m > ad_max_amp: ad_max_amp = m

    def scale_from_max(max_amp):
        if not np.isfinite(max_amp) or max_amp == 0.0:
            return 0, 1.0  # e=0, scale=1
        _, e = np.frexp(max_amp)          # max_amp ≈ mantissa * 2**e
        s = np.ldexp(1.0, -e)             # = 2**(-e)
        return e, s

    e_ra, s_ra = scale_from_max(ra_max_amp)
    e_rd, s_rd = scale_from_max(rd_max_amp)
    e_ad, s_ad = scale_from_max(ad_max_amp)

    # ---------- Pass 1: 在缩放域里累计 sum / sumsq ----------
    # 统计项（sum / sumsq / N），使用 longdouble 防溢出；N 用 Python int
    ra = {"sum_r": np.longdouble(0.0), "sumsq_r": np.longdouble(0.0),
          "sum_i": np.longdouble(0.0), "sumsq_i": np.longdouble(0.0),
          "sum_mag": np.longdouble(0.0), "sumsq_mag": np.longdouble(0.0),
          "sum_pha": np.longdouble(0.0), "sumsq_pha": np.longdouble(0.0),
          "N": 0}
    rd = {"sum_r": np.longdouble(0.0), "sumsq_r": np.longdouble(0.0),
          "sum_i": np.longdouble(0.0), "sumsq_i": np.longdouble(0.0),
          "sum_mag": np.longdouble(0.0), "sumsq_mag": np.longdouble(0.0),
          "sum_pha": np.longdouble(0.0), "sumsq_pha": np.longdouble(0.0),
          "N": 0}
    ad = {"sum_r": np.longdouble(0.0), "sumsq_r": np.longdouble(0.0),
          "sum_i": np.longdouble(0.0), "sumsq_i": np.longdouble(0.0),
          "sum_mag": np.longdouble(0.0), "sumsq_mag": np.longdouble(0.0),
          "sum_pha": np.longdouble(0.0), "sumsq_pha": np.longdouble(0.0),
          "N": 0}

    def accumulate(domain_dict, real, imag, mag, pha, scale):
        # 统一用相同 scale（来自最大幅度）缩放实部/虚部/模；相位不缩放
        r = (real * scale).astype(np.longdouble, copy=False)
        i = (imag * scale).astype(np.longdouble, copy=False)
        m = (mag  * scale).astype(np.longdouble, copy=False)
        p = pha.astype(np.longdouble, copy=False)  # [-pi, pi]，无需缩放

        # add.reduce 更稳定，且 dtype=longdouble
        domain_dict["sum_r"]   += np.add.reduce(r, dtype=np.longdouble)
        domain_dict["sumsq_r"] += np.add.reduce(r*r, dtype=np.longdouble)
        domain_dict["sum_i"]   += np.add.reduce(i, dtype=np.longdouble)
        domain_dict["sumsq_i"] += np.add.reduce(i*i, dtype=np.longdouble)
        domain_dict["sum_mag"] += np.add.reduce(m, dtype=np.longdouble)
        domain_dict["sumsq_mag"] += np.add.reduce(m*m, dtype=np.longdouble)
        domain_dict["sum_pha"] += np.add.reduce(p, dtype=np.longdouble)
        domain_dict["sumsq_pha"] += np.add.reduce(p*p, dtype=np.longdouble)
        domain_dict["N"] += real.size  # Python int，避免回绕

    for idx in tqdm(trainIdx, desc=f"[Pass1] accumulate {use_cluster_filter}"):
        seqPath = os.path.join(root, f'uav_seqs_{idx}', f'{trainDataType}_slice_frame_{use_cluster_filter}', radarType)
        raPath = os.path.join(seqPath, 'raw_frame_RA')
        rdPath = os.path.join(seqPath, 'raw_frame_RD')
        adPath = os.path.join(seqPath, 'raw_frame_AD')

        if not only_RD_AD:
            for fn in list_files(raPath):
                real, imag, mag, pha, comp = load_pair(os.path.join(raPath, fn), trainDataType)
                if real.size == 0: continue
                accumulate(ra, real, imag, mag, pha, s_ra)

        for fn in list_files(rdPath):
            real, imag, mag, pha, comp = load_pair(os.path.join(rdPath, fn), trainDataType)
            if real.size == 0: continue
            accumulate(rd, real, imag, mag, pha, s_rd)

        for fn in list_files(adPath):
            real, imag, mag, pha, comp = load_pair(os.path.join(adPath, fn), trainDataType)
            if real.size == 0: continue
            accumulate(ad, real, imag, mag, pha, s_ad)

    def finalize(domain_dict, e, scale):
        """
        从缩放域 sum/sumsq 还原出原始域的均值/标准差。
        返回：mean_r, std_r, mean_i, std_i, mean_mag, std_mag, mean_pha, std_pha
        """
        N = domain_dict["N"]
        if N == 0:
            z = (np.nan,)*8
            return z

        N_ld = np.longdouble(N)

        # 缩放域统计
        mu_r_s  = domain_dict["sum_r"] / N_ld
        mu_i_s  = domain_dict["sum_i"] / N_ld
        mu_mag_s = domain_dict["sum_mag"] / N_ld
        mu_pha   = domain_dict["sum_pha"] / N_ld  # 相位未缩放

        # var = E[x^2] - (E[x])^2
        var_r_s  = domain_dict["sumsq_r"] / N_ld - mu_r_s * mu_r_s
        var_i_s  = domain_dict["sumsq_i"] / N_ld - mu_i_s * mu_i_s
        var_mag_s = domain_dict["sumsq_mag"] / N_ld - mu_mag_s * mu_mag_s
        var_pha   = domain_dict["sumsq_pha"] / N_ld - mu_pha * mu_pha

        # 数值护栏
        var_r_s  = max(var_r_s, 0.0)
        var_i_s  = max(var_i_s, 0.0)
        var_mag_s = max(var_mag_s, 0.0)
        var_pha   = max(var_pha, 0.0)

        # 还原到原始域：均值乘 2**e，标准差乘 2**e
        mu_r   = np.ldexp(mu_r_s, e)
        mu_i   = np.ldexp(mu_i_s, e)
        mu_mag = np.ldexp(mu_mag_s, e)
        std_r  = np.ldexp(np.sqrt(var_r_s), e)
        std_i  = np.ldexp(np.sqrt(var_i_s), e)
        std_mag = np.ldexp(np.sqrt(var_mag_s), e)

        std_pha = float(np.sqrt(var_pha))  # 相位不缩放

        return (float(mu_r), float(std_r),
                float(mu_i), float(std_i),
                float(mu_mag), float(std_mag),
                float(mu_pha), float(std_pha))

    # 计算三类统计
    if not only_RD_AD:
        (ra_mu_r, ra_std_r,
         ra_mu_i, ra_std_i,
         ra_mu_mag, ra_std_mag,
         ra_mu_pha, ra_std_pha) = finalize(ra, e_ra, s_ra)
        # 复均值 = mu_r + j*mu_i
        ra_mu_complex = complex(ra_mu_r, ra_mu_i)
        # 复标准差（按你原先的定义）：sqrt(Var(real) + Var(imag))
        ra_std_complex = float(np.sqrt(ra_std_r**2 + ra_std_i**2))
        ra_N = ra["N"]
    else:
        ra_mu_r = ra_std_r = ra_mu_i = ra_std_i = ra_mu_mag = ra_std_mag = ra_mu_pha = ra_std_pha = np.nan
        ra_mu_complex = complex(np.nan, np.nan)
        ra_std_complex = np.nan
        ra_N = 0

    (rd_mu_r, rd_std_r,
     rd_mu_i, rd_std_i,
     rd_mu_mag, rd_std_mag,
     rd_mu_pha, rd_std_pha) = finalize(rd, e_rd, s_rd)
    rd_N = rd["N"]

    (ad_mu_r, ad_std_r,
     ad_mu_i, ad_std_i,
     ad_mu_mag, ad_std_mag,
     ad_mu_pha, ad_std_pha) = finalize(ad, e_ad, s_ad)
    ad_N = ad["N"]

    # ---------- 写结果 ----------
    os.makedirs(os.path.dirname("./temp.txt"), exist_ok=True)
    if not os.path.exists("./temp.txt"):
        with open("./temp.txt", "w") as f:
            pass

    with open("./temp.txt", "a") as f:
        f.write(f"{use_cluster_filter}\n")
        f.write(f"{trainDataType}_{use_cluster_filter}\n")

        if not only_RD_AD:
            f.write("RA:\n")
            f.write(f"实部均值={ra_mu_r}\n实部标准差={ra_std_r}\n")
            f.write(f"虚部均值={ra_mu_i}\n虚部标准差={ra_std_i}\n")
            f.write(f"模均值={ra_mu_mag}\n模标准差={ra_std_mag}\n")
            f.write(f"相位均值={ra_mu_pha}\n相位标准差={ra_std_pha}\n")
            f.write(f"复数实部均值={ra_mu_complex.real}\n复数虚部均值={ra_mu_complex.imag}\n标准差={ra_std_complex}\n")
            f.write(f"N={ra_N}\n")

        f.write("RD:\n")
        f.write(f"实部均值={rd_mu_r}\n实部标准差={rd_std_r}\n")
        f.write(f"虚部均值={rd_mu_i}\n虚部标准差={rd_std_i}\n")
        f.write(f"模均值={rd_mu_mag}\n模标准差={rd_std_mag}\n")
        f.write(f"相位均值={rd_mu_pha}\n相位标准差={rd_std_pha}\n")
        f.write(f"N={rd_N}\n")

        f.write("AD:\n")
        f.write(f"实部均值={ad_mu_r}\n实部标准差={ad_std_r}\n")
        f.write(f"虚部均值={ad_mu_i}\n虚部标准差={ad_std_i}\n")
        f.write(f"模均值={ad_mu_mag}\n模标准差={ad_std_mag}\n")
        f.write(f"相位均值={ad_mu_pha}\n相位标准差={ad_std_pha}\n")
        f.write(f"N={ad_N}\n")
