import os

root = "/home/jackychou/code/RODNet_UAV/workers"

file_list = [
    'rodnet-HGwI-single-4_128_11_baseline_r&i',
]

for file_path in file_list:
    print(file_path)
    file_root = os.path.join(root, file_path)
    file_root = os.path.join(file_root, os.listdir(file_root)[0])
    log_path = os.path.join(file_root, 'train.log')
    best_epoch = 0
    best_score = 0
    best_ap50 = 0
    best_ap70 = 0
    best_map = 0
    with open(log_path, 'r') as f:
        epoch = 6
        for line in f:
            if 'Overall            |' in line:
                line = line.split('|')
                mean_ap = float(line[1].split(":")[-1])
                ap50 = float(line[2].split(":")[-1])
                ap70 = float(line[3].split(":")[-1])
                score = ap50 * 0.4 + mean_ap * 0.4 + ap70 * 0.2
                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    best_ap50 = ap50
                    best_ap70 = ap70
                    best_map = mean_ap
                epoch = epoch + 1
    print(best_epoch, best_score, best_ap50, best_ap70, best_map)