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
