import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import time



def check_path(path):
    if isinstance(path, str):
        path = [path]
    for p in path:
        os.makedirs(p, exist_ok=True)


def process_one_frame(args):
    idx, frame_idx = args
    seqPath = os.path.join(data_root, f'uav_seqs_{idx}')
    RA_Path = os.path.join(
        seqPath, f'python_slice_frame_{filter_type}', 'azimuth', 'raw_frame_RA'
    )
    saveRAPath = os.path.join(
        save_root,
        f'uav_seqs_{idx}',
        f'python_slice_frame_{filter_type}_PCAv1',
        'azimuth',
        'raw_frame_RA'
    )

    os.makedirs(saveRAPath, exist_ok=True)
    temp_RA_list = []
    for chirp_idx in range(chirp_num):
        temp_RA_path = os.path.join(
            RA_Path, f"{frame_idx:03d}_{chirp_idx + 1:09d}.npy"
        )
        temp_sub_RA = np.load(temp_RA_path)  # (..., 2)
        temp_sub_RA = temp_sub_RA[..., 0] + temp_sub_RA[..., 1] * 1j
        temp_RA_list.append(temp_sub_RA)

    temp_RA = np.stack(temp_RA_list, axis=0)
    c, r, a = temp_RA.shape
    temp_RA = np.transpose(temp_RA.reshape(c, r * a), (1, 0))
    temp_RA_centered = temp_RA - np.mean(temp_RA, axis=0)
    U, S, Vh = np.linalg.svd(temp_RA_centered, full_matrices=False)
    explained_variance = (S ** 2) / (len(temp_RA_centered) - 1)
    explained_variance_ratio = explained_variance / np.sum(explained_variance, dtype=np.float128)

    if idx in trainIdx:
        r1   = float(np.sum(explained_variance_ratio[:1], dtype=np.float128))
        r4   = float(np.sum(explained_variance_ratio[:4], dtype=np.float128))
        r8   = float(np.sum(explained_variance_ratio[:8], dtype=np.float128))
        r16  = float(np.sum(explained_variance_ratio[:16], dtype=np.float128))
        r32  = float(np.sum(explained_variance_ratio[:32], dtype=np.float128))
        r64  = float(np.sum(explained_variance_ratio[:64], dtype=np.float128))
        r128 = float(np.sum(explained_variance_ratio[:128], dtype=np.float128))
    else:
        r1 = None
        r4 = None
        r8 = None
        r16 = None
        r32 = None
        r64 = None
        r128 = None

    k = PCA_size
    V_k = Vh[:k, ...]  # (k, c)
    temp_RA_PCA = temp_RA_centered @ V_k.conj().T  # (r*a, k)

    temp_RA_PCA = np.concatenate(
        (
            np.expand_dims(np.real(temp_RA_PCA).reshape(r, a, PCA_size), axis=-1),
            np.expand_dims(np.imag(temp_RA_PCA).reshape(r, a, PCA_size), axis=-1)
        ),
        axis=-1
    )

    for save_idx in range(temp_RA_PCA.shape[-2]):  # k
        sub_temp_RA_PCA = temp_RA_PCA[..., save_idx, :]
        sub_temp_RA_PCA = sub_temp_RA_PCA
        sub_save_name = os.path.join(
            saveRAPath, f"{frame_idx:03d}_{save_idx + 1:09d}.npy"
        )
        np.save(sub_save_name, sub_temp_RA_PCA)

    return r1, r4, r8, r16, r32, r64, r128


data_root = "/home/jackychou/Zhou/dataset/UAVRadar"
save_root = "/home/jackychou/Zhou/dataset/UAVRadar"
filter_type = 11
PCA_size = 128
start_idx = 1
end_idx = 79
frame_num = 300
chirp_num = 255
trainIdx = [1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62, 64, 66, 67, 68, 69, 70, 72, 75, 76, 78, 79]
all_Idx = [i for i in range(start_idx, end_idx + 1)]
# all_Idx = [i for i in all_Idx_a if i not in set(trainIdx)]
num_workers = 8

if __name__ == "__main__":
    t0 = time.time()
    for idx in all_Idx:
        saveSeqPath = os.path.join(save_root, f'uav_seqs_{idx}')
        check_path(os.path.join(saveSeqPath, f'python_slice_frame_{filter_type}_PCAv1'))
        check_path(os.path.join(saveSeqPath, f'python_slice_frame_{filter_type}_PCAv1', 'azimuth'))
        check_path(os.path.join(saveSeqPath, f'python_slice_frame_{filter_type}_PCAv1', 'azimuth', 'raw_frame_RA'))

    tasks = []
    for idx in all_Idx:
        for frame_idx in range(frame_num):
            tasks.append((idx, frame_idx))

    ratio1 = []
    ratio4 = []
    ratio8 = []
    ratio16 = []
    ratio32 = []
    ratio64 = []
    ratio128 = []

    with mp.Pool(processes=num_workers) as pool:
        # 用 imap_unordered + tqdm 显示整体进度条
        for r1, r4, r8, r16, r32, r64, r128 in tqdm(
            pool.imap_unordered(process_one_frame, tasks),
            total=len(tasks)
        ):
            if r1 is not None:
                ratio1.append(r1)
                ratio4.append(r4)
                ratio8.append(r8)
                ratio16.append(r16)
                ratio32.append(r32)
                ratio64.append(r64)
                ratio128.append(r128)

    print("######################################")
    print(f"1:   {np.mean(np.array(ratio1))}")
    print(f"4:   {np.mean(np.array(ratio4))}")
    print(f"8:   {np.mean(np.array(ratio8))}")
    print(f"16:  {np.mean(np.array(ratio16))}")
    print(f"32:  {np.mean(np.array(ratio32))}")
    print(f"64:  {np.mean(np.array(ratio64))}")
    print(f"128:  {np.mean(np.array(ratio128))}")
    print("######################################")

    print(f"Total time: {time.time() - t0:.2f} s")
