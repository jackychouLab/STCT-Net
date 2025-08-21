from config_dataset_128 import *

model_cfg = dict(
    type='E_RODNet',
    name='E_RODNet',
    max_dets=5,
    peak_thres=0.15,
    ols_thres=0.3,
    mnet_cfg=(32, 32),
    dcn=False,
    train_type='single',
)

train_cfg = dict(
    batch_size=4,
    win_size=16,
    train_stride=4,
    log_step=50,
    train_step=1,
    seed=2027,
    num_workers=8,
    eval_epoch_list=[i for i in range(6, 56)],
    use_aug=False,
    use_mix_aug=False,
    use_filter=11,
    norm_type='real&image',
    loss_type='smooth_l1',
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