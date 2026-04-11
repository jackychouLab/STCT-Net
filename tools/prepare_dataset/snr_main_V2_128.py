import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import scipy.constants

# 雷达默认参数
radar_config = {}
radar_config['c'] = scipy.constants.speed_of_light
radar_config['fc'] = 60.25e9
radar_config['lambda'] = radar_config['c'] / radar_config['fc']
radar_config['Tx'] = 3
radar_config['Rx'] = 4

radar_config['Fs'] = 6.25e6
radar_config['sweepSlope'] = 9.994e12
radar_config['samples'] = 512
radar_config['loop'] = 255

radar_config['Tc'] = 354e-6
radar_config['fft_Range'] = 512 + 24 # 134-->128
radar_config['fft_Velcity'] = 256
radar_config['fft_Angle'] = 128
radar_config['num_crop'] = 12
radar_config['max_value'] = 1e4

radar_config['duration'] = 30
radar_config['frame_number'] = 10
radar_config['Lanes'] = 2

radar_config['ramap_rsize_label'] = 512
radar_config['ramap_asize_label'] = 128

radar_config['ra_min_label'] = -60
radar_config['ra_max_label'] = 60

freq_res = radar_config['Fs'] / radar_config['fft_Range']
freq_grid = np.arange(radar_config['fft_Range']) * freq_res
radar_config['rng_grid'] = freq_grid * radar_config['c'] / radar_config['sweepSlope'] / 2


w = np.linspace(-1, 1, radar_config['fft_Angle'])
radar_config['agl_grid'] = np.arcsin(w)

radar_config['vel_grid'] = 1 / radar_config['Tc'] / radar_config['loop'] * np.arange(-256 / 2, 256 /2) * radar_config['c'] / (2 * radar_config['fc'])


radar_config['ramap_rsize'] = 512
radar_config['ramap_asize'] = 128
radar_config['ramap_vsize'] = 256
radar_config['ramap_esize'] = 128


radar_config['rr_min'] = radar_config['rng_grid'][radar_config['num_crop']]
radar_config['rr_max'] = radar_config['rng_grid'][-radar_config['num_crop'] - 1]

radar_config['ra_min'] = -90
radar_config['ra_max'] = 90

radar_config['class_table'] = {0:"background", 1:"uav"}
radar_config['confmap_sigmas'] = {'uav': 15}
radar_config['confmap_sigmas_interval'] = {'uav': [7, 15]}
radar_config['confmap_length'] = {'uav': 1}
radar_config['object_sizes'] = {'uav': 1.0}


"""
更改导入包/文件路径
只要更改一下root路径直接运行即可，会在uav_seqs_n文件夹下面创建两个文件夹：
- RTK_label：距离角度标签
- SNR_Result
    - snr_result：snr结果，txt格式
    - visualization：snr结果，可视化

‘每一帧用一个txt文件记录’
"""


def check_path(x):
    if isinstance(x, str):
        x = [x]
    for x_i in x:
        if not os.path.exists(x_i):
            os.makedirs(x_i)


def get_sim_idx(input_key, query_list):
    return np.abs(np.array(query_list) - input_key).argmin()


class SnrObject:
    def __init__(self):
        self.root = r"/home/jackychou/dataset/UAV1.0"
        self.numCrop = 128
        self.saveroot = '/mnt/c/Ubuntu-temp/mmUAV128/txtData'
        self.start_single_id = 1
        self.end_single_id = 79
        self.window_w = 9 # 9
        self.window_h = 3 # 3
        self.num_frames = 300
        self.numChirps = 32
        self.rng_grid = radar_config['rng_grid']
        self.agl_grid = radar_config['agl_grid']
        self.initialize()
        self.Visualization = False

    def initialize(self):
        for i in range(self.start_single_id, self.end_single_id + 1):
            radar_data_dir = os.path.join(self.root, f"uav_seqs_{i}", "python_slice_frame_1132", "azimuth", "raw_frame_RA")
            label_dir = os.path.join(self.saveroot, f"uav_seqs_{i}", f"RTK_label_{self.numCrop}", "azimuth")
            save_dir = os.path.join(self.saveroot, f"uav_seqs_{i}", "SNR_Result", "azimuth", f'snr_result_{self.numCrop}')
            rgb_dir = os.path.join(self.saveroot, f"uav_seqs_{i}", "SNR_Result", "azimuth", f'visualization_{self.numCrop}')
            check_path([save_dir, label_dir, rgb_dir])
            setattr(self, f'radarDataDir{i}', radar_data_dir)
            setattr(self, f'labelDir{i}', label_dir)
            setattr(self, f'saveDir{i}', save_dir)
            setattr(self, f'rgbDir{i}', rgb_dir)

    def process_frame(self, frame_idx, radar_data_dir, label_dir, save_dir, rgb_dir):
        # Read and accumulate data from all chirps
        all_arr = None
        for z in range(1, self.numChirps + 1):
            file_name = f"{frame_idx:03d}_{z:09d}.npy"
            file_path = os.path.join(radar_data_dir, file_name)
            try:
                arr = np.load(file_path)
                complex_array = arr[..., 0] + 1j * arr[..., 1]
                magnitude_array = np.abs(complex_array)[..., np.newaxis]
                if all_arr is None:
                    all_arr = np.zeros_like(magnitude_array)
                all_arr += magnitude_array
            except Exception as e:
                print(f"Error loading or processing {file_path}: {e}")

        all_arr = np.sum(all_arr, axis=-1) / self.numChirps

        # Get labels
        label_file = os.path.join(label_dir, f"{frame_idx:09d}.txt")
        angles, ranges = self.get_label(label_file)

        # Calculate SNR
        signal_y1 = max(0, int(ranges[0]) - self.window_h)
        signal_y2 = min(all_arr.shape[0], int(ranges[0]) + self.window_h + 1)
        signal_x1 = max(0, int(angles[0]) - self.window_w)
        signal_x2 = min(all_arr.shape[1], int(angles[0]) + self.window_w + 1)

        signal = np.sum(all_arr[signal_y1:signal_y2, signal_x1:signal_x2])
        noise = np.sum(all_arr) - signal

        signal /= ((signal_y2 - signal_y1) * (signal_x2 - signal_x1))
        noise /= all_arr.size - (signal_y2 - signal_y1) * (signal_x2 - signal_x1)

        snr = 20 * np.log10(signal / noise) if noise > 0 else float('inf')

        # Write SNR to text file
        with open(os.path.join(save_dir, f"{frame_idx:09d}.txt"), 'w') as f:
            f.write(f"{snr}")

        if self.Visualization:
            # Visualization
            plt.figure(figsize=(10, 5))
            plt.imshow(all_arr, cmap='viridis', origin='lower')
            plt.scatter(angles, ranges, color='red', s=2)
            plt.title('RA')
            ax = plt.gca()
            plt.text(0.99, 0.01, f'SNR: {snr:.2f}',
                     verticalalignment='bottom', horizontalalignment='right',
                     transform=ax.transAxes,
                     color='white', fontsize=10,
                     bbox={'alpha': 0, 'pad': 5, 'edgecolor': 'none'})
            plt.savefig(os.path.join(rgb_dir, f"{frame_idx:09d}.png"), dpi=300, transparent=True, bbox_inches='tight')
            plt.close()

        return snr

    def get_label(self, path):
        xs, ys = [], []
        with open(path, 'r') as f:
            for line in f:
                x = float(line.split()[0])
                y = float(line.split()[1])
                xs.append(x)
                ys.append(y)
        return xs, ys


    def processRadarDataSnr(self):
        for seq_id in range(self.start_single_id, self.end_single_id + 1):
            radar_data_dir = getattr(self, f'radarDataDir{seq_id}')
            label_dir = getattr(self, f'labelDir{seq_id}')
            save_dir = getattr(self, f'saveDir{seq_id}')
            rgb_dir = getattr(self, f'rgbDir{seq_id}')

            # Using a process pool to parallelize frame processing
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.process_frame, j, radar_data_dir, label_dir, save_dir, rgb_dir) for j in
                           range(self.num_frames)]
                snr_results = [future.result() for future in
                               tqdm(futures, total=self.num_frames, desc=f"Processing sequence {seq_id}")]

            with open(os.path.join(save_dir, f"mean_snr.txt"), 'w') as f:
                f.write(f"{np.mean(snr_results)}")
                f.close()


if __name__ == "__main__":
    snrObject = SnrObject()

    # Calculate SNR
    snrObject.processRadarDataSnr()
