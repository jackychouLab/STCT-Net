from config_dataset_128 import *

model_cfg = dict(
    type='myNet',
    name='myNet',
    max_dets=5,
    peak_thres=0.3,
    ols_thres=0.3,
    mnet_cfg=(32, 32),
    mnet_type=1,        # STCTM 5; MNet-Max 1; MNet-Avg 3; E-RODNet 2
    train_type='single', # single or multi
    head_size=(3, 3, 3),
    norm_type='gn',
    act_type='gelu',
    full_conv=True,
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
    use_filter=1132,
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
