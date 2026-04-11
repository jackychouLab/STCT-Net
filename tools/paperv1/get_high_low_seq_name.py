import numpy as np
from snr_list import all_snr_origin_512

grouped_all_snr_origin_512 = [
    all_snr_origin_512[i:i+300] for i in range(0, len(all_snr_origin_512), 300)
]
groups = np.array(grouped_all_snr_origin_512)
groups = np.mean(groups, axis=1)

train_idx = [1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
              33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62, 64,
              66, 67, 68, 69, 70, 72, 75, 76, 78, 79]

val_idx = [2, 6, 7, 8, 37, 45, 47, 55, 59, 63, 65, 71, 73, 74, 77]

mean_snr = 0.
for train_id in train_idx:
    mean_snr += groups[train_id-1]
mean_snr /= len(train_idx)
print(mean_snr)

for val_id in val_idx:
    if groups[val_id-1] >= mean_snr:
        continue
    else:
        print(val_id)

# mean SNR 2.23

# high 2 6 7 8 37 45 47 55 59 63 73

# low 65 71 74 77