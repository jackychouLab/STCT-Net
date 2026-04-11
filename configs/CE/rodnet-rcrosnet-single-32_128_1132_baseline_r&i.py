from configs.config_dataset_128 import *

model_cfg = dict(
    type='RC-ROSNet',
    name='RC-ROSNet',
    max_dets=5,
    peak_thres=0.3,
    ols_thres=0.3,
    train_type='single',
    mnet_cfg=(32, 0),
)

train_cfg = dict(
    batch_size=4,
    win_size=1,
    train_stride=1,
    log_step=200,
    train_step=1,
    seed=2027,
    num_workers=8,
    eval_epoch_list=[i for i in range(6, 56)],
    use_filter=1132,
    norm_type='real&image',
    loss_type='mse',
    )

optim_cfg = dict(
    type='adamw',
    lr=0.0001,
)

schedule_cfg = dict(
    type='Cos',
    n_epoch=55,
    warmup_epoch=5,
)

test_cfg = dict(
    test_step=1,
    test_stride=1,
    rr_min=radar_config['rr_min'],
    rr_max=radar_config['rr_max'],
    ra_min=radar_config['ra_min'],
    ra_max=radar_config['ra_max'],
)
