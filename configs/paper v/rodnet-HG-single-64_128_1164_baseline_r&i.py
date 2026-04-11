from config_dataset_128_2 import *

model_cfg = dict(
    type='HGv2',
    name='rodnet-hg1v2-win16-mnet',
    max_dets=5,
    peak_thres=0.3,
    ols_thres=0.3,
    stacked_num=1,
    mnet_cfg=(64, 32),
    dcn=False,
    train_type='single',
)

train_cfg = dict(
    batch_size=4,
    win_size=16,
    train_stride=4,
    log_step=50,
    train_step=1,
    seed=42,
    num_workers=8,
    eval_epoch_list=[i for i in range(5, 61)],
    use_aug=False,
    use_mix_aug=False,
    use_filter=1164,
    norm_type='real&image',
    loss_type='bce',
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
