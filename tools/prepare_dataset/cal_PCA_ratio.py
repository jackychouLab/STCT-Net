import os
import numpy as np
from tqdm import tqdm
from scipy.linalg import svd
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat

np.random.seed(2027)


def check_path(x):
    if isinstance(x, str):
        x = [x]
    for xi in x:
        if not os.path.exists(xi):
            os.mkdir(xi)


data_root = "/dssg/home/dz_hzj/ZJH/dataset/Radar-UAV"
save_root = "/dssg/home/dz_hzj/ZJH/dataset/Radar-UAV"
filter_type = 11
PCA_size = 128   # 这里其实对应你要看的最大主成分数
trainIdx = [1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 66, 67, 68, 69, 70, 72, 75, 76, 78, 79]
frame_num = 300
chirp_num = 255

MAX_WORKERS = 12


def process_frame(frame_idx, RA_Path, chirp_num, PCA_size):
    """
    处理单个 frame：
    - 读取该 frame 的所有 chirp 的 RA（复数形式）
    - 做 SVD/PCA
    - 计算 1/4/8/16/32/64/128 个主成分的累计方差占比
    返回 7 个 ratio
    """
    # 1. 读取所有 chirp 的 RA 数据，堆成 [c, r, a]
    ra_list = []
    for chirp_idx in range(chirp_num):
        temp_RA_path = os.path.join(RA_Path, f"{frame_idx:03d}_{chirp_idx+1:09d}.npy")
        temp_sub_RA = np.load(temp_RA_path)
        temp_sub_RA = temp_sub_RA.astype(np.float64, copy=False)
        temp_complex = temp_sub_RA[..., 0] + 1j * temp_sub_RA[..., 1]
        ra_list.append(temp_complex)

    # (c, r, a)
    temp_RA = np.stack(ra_list, axis=0)

    # 2. PCA：reshape + 居中 + SVD
    c, r, a = temp_RA.shape
    temp_RA = np.transpose(temp_RA.reshape(c, r * a), (1, 0))  # [r*a, c]
    temp_RA_centered = temp_RA - np.mean(temp_RA, axis=0, keepdims=True)

    U, S, Vh = svd(temp_RA_centered, full_matrices=False)

    # 3. 解释方差和比例（尽量用 float128 做求和，减小误差）
    n_samples = temp_RA_centered.shape[0]
    S2 = S.astype(np.float128) ** 2
    explained_variance = S2 / np.float128(n_samples - 1)
    total_var = np.sum(explained_variance, dtype=np.float128)
    explained_variance_ratio = explained_variance / total_var

    def ratio_for(k):
        k = min(k, explained_variance_ratio.shape[0])
        return np.sum(explained_variance_ratio[:k], dtype=np.float128)

    r1 = ratio_for(1)
    r4 = ratio_for(4)
    r8 = ratio_for(8)
    r16 = ratio_for(16)
    r32 = ratio_for(32)
    r64 = ratio_for(64)
    r128 = ratio_for(128)

    return r1, r4, r8, r16, r32, r64, r128


ratio1 = []
ratio4 = []
ratio8 = []
ratio16 = []
ratio32 = []
ratio64 = []
ratio128 = []

# 计算均值（并行按 frame）
for idx in trainIdx:
    seqPath = os.path.join(data_root, f'uav_seqs_{idx}')
    RA_Path = os.path.join(seqPath, f'python_slice_frame_{filter_type}', 'azimuth', 'raw_frame_RA')
    saveSeqPath = os.path.join(save_root, f'uav_seqs_{idx}')

    # create save path（虽然这里没保存 PCA 结果，但保持和原代码一致）
    check_path(os.path.join(saveSeqPath, f'python_slice_frame_{filter_type}_PCAv1'))
    check_path(os.path.join(saveSeqPath, f'python_slice_frame_{filter_type}_PCAv1', 'azimuth'))
    check_path(os.path.join(saveSeqPath, f'python_slice_frame_{filter_type}_PCAv1', 'azimuth', 'raw_frame_RA'))

    frame_indices = list(range(frame_num))

    from itertools import repeat
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        iterator = executor.map(
            process_frame,
            frame_indices,
            repeat(RA_Path),
            repeat(chirp_num),
            repeat(PCA_size),
        )
        for r1, r4, r8, r16, r32, r64, r128 in tqdm(
            iterator,
            total=frame_num,
            desc=f"Seq {idx} PCA ratio (filter={filter_type})",
            dynamic_ncols=True
        ):
            ratio1.append(r1)
            ratio4.append(r4)
            ratio8.append(r8)
            ratio16.append(r16)
            ratio32.append(r32)
            ratio64.append(r64)
            ratio128.append(r128)

# 写结果
out_path = "/dssg/home/dz_hzj/ZJH/code/mean_std/ratio.txt"
if not os.path.exists(out_path):
    with open(out_path, "w") as f:
        pass

with open(out_path, "a") as f:
    f.write("######################################\n")
    f.write(f"1:{np.mean(np.array(ratio1, dtype=np.float128))}\n")
    f.write(f"4:{np.mean(np.array(ratio4, dtype=np.float128))}\n")
    f.write(f"8:{np.mean(np.array(ratio8, dtype=np.float128))}\n")
    f.write(f"16:{np.mean(np.array(ratio16, dtype=np.float128))}\n")
    f.write(f"32:{np.mean(np.array(ratio32, dtype=np.float128))}\n")
    f.write(f"64:{np.mean(np.array(ratio64, dtype=np.float128))}\n")
    f.write(f"128:{np.mean(np.array(ratio128, dtype=np.float128))}\n")
    f.write("######################################\n")
