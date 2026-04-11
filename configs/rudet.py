from configs.UAV_60_radar import radar_config



radar_configs = {
    'ramap_rsize': radar_config['ramap_rsize'],  # RVAEMap range size
    'ramap_vsize': radar_config['ramap_vsize'],  # RVAEMap velcity size
    'ramap_asize': radar_config['ramap_asize'],  # RVAEMap azimth size
    'ramap_esize': radar_config['ramap_esize'],  # RVAEMap elevation size
    'frame_rate': radar_config['frame_number'],
    'crop_num': radar_config['num_crop'],
    'n_chirps': radar_config['loop'],
    'sample_freq': radar_config['Fs'],
    'sweep_slope': radar_config['sweepSlope'],
    'ramap_rsize_label': radar_config['ramap_rsize_label'],
    'ramap_asize_label': radar_config['ramap_asize_label'],
    'ra_min_label': radar_config['ra_min_label'],  # min radar angle
    'ra_max_label': radar_config['ra_max_label'],  # max radar angle
    'rr_min': radar_config['rr_min'],              # min radar range (fixed)
    'rr_max': radar_config['rr_max'],              # max radar range (fixed)
    'ra_min': radar_config['ra_min'],              # min radar angle (fixed)
    'ra_max': radar_config['ra_max'],              # max radar angle (fixed)
}