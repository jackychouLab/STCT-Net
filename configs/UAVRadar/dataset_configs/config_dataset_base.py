from configs.UAVRadar.dataset_configs.UAV_60_radar import radar_config

dataset_cfg = dict(
    dataset_name='mmUAV',
    base_root="/home/jackychou/dataset/UAVRadar",
    data_root="/home/jackychou/dataset/UAVRadar/uav_seqs_{}",
    anno_root="/home/jackychou/dataset/UAVRadar/uav_seqs_{}/annot",
    anno_ext='.csv',
    train=dict(
        subdir='train',
        seqs=[1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
              33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62, 64,
              66, 67, 68, 69, 70, 72, 75, 76, 78, 79],
    ),
    valid=dict(
        subdir='valid',
        seqs=[2, 6, 7, 8, 37, 45, 47, 55, 59, 63, 65, 71, 73, 74, 77],
    ),
    test=dict(
        subdir='test',
        seqs=[2, 6, 7, 8, 37, 45, 47, 55, 59, 63, 65, 71, 73, 74, 77],
    ),
    demo=dict(
        subdir='demo',
        seqs=[],
    ),
    data_mean_std=dict(

        RA11_real_mean=6.903373706013127e-11,
        RA11_real_std=1012.9295043945312,
        RA11_imag_mean=5.342893488746725e-11,
        RA11_imag_std=1012.9020385742188,
        RD11_real_mean=-0.45914414525032043,
        RD11_real_std=1531.5933837890625,
        RD11_imag_mean=-0.5188534259796143,
        RD11_imag_std=1529.7945556640625,
        AD11_real_mean=-0.5543321967124939,
        AD11_real_std=459.515625,
        AD11_imag_mean=3.830321788787842,
        AD11_imag_std=459.5393371582031,
        RD1132_real_mean=-0.45914414525032043,
        RD1132_real_std=1531.5933837890625,
        RD1132_imag_mean=-0.5188534259796143,
        RD1132_imag_std=1529.7945556640625,
        AD1132_real_mean=-0.5543321967124939,
        AD1132_real_std=459.515625,
        AD1132_imag_mean=3.830321788787842,
        AD1132_imag_std=459.5393371582031,

        RA114_real_mean=2.874871229963105e-09,
        RA114_real_std=2905.1337890625,
        RA114_imag_mean=3.905180889773874e-08,
        RA114_imag_std=2904.929931640625,

        RA118_real_mean=1.0049455489991033e-08,
        RA118_real_std=2454.12158203125,
        RA118_imag_mean=2.5085540755753755e-08,
        RA118_imag_std=2453.905029296875,

        RA1116_real_mean=4.02903799212595e-09,
        RA1116_real_std=2118.89697265625,
        RA1116_imag_mean=1.4310892026969668e-08,
        RA1116_imag_std=2118.724609375,

        RA1132_real_mean=-3.20246518192846e-09,
        RA1132_real_std=1838.5626220703125,
        RA1132_imag_mean=4.413432730387967e-09,
        RA1132_imag_std=1838.4454345703125,

        RA1164_real_mean=-3.8616487785247955e-09,
        RA1164_real_std=1560.6644287109375,
        RA1164_imag_mean=7.0037011745682776e-09,
        RA1164_imag_std=1560.6119384765625,

        RA11128_real_mean=-7.926763412324789e-12,
        RA11128_real_std=1280.9857177734375,
        RA11128_imag_mean=8.430787623581182e-09,
        RA11128_imag_std=1280.95458984375,
    ),
    rangeDownSample=1,
)

confmap_cfg = dict(
    confmap_sigmas={
        'uav': 10, # Z. Jiang:10     J. Zhou:5     GPT:20
    },
    confmap_sigmas_interval={
        'uav': [5, 10], # Z. Jiang:[5, 10]   J. Zhou:[2, 6]     GPT:[8, 20]
    },
    confmap_length={
        'uav': 1,
    }
)

test_cfg = dict(
    test_step=1,
    test_stride=4,
    rr_min=radar_config['rr_min'],
    rr_max=radar_config['rr_max'],
    ra_min=radar_config['ra_min'],
    ra_max=radar_config['ra_max'],
)