from config_dataset_128 import *

model_cfg = dict(
    type='RAMP',
    name='RAMP',
    max_dets=5,
    peak_thres=0.3,
    ols_thres=0.3,
    train_type='single',
)

train_cfg = dict(
    batch_size=4,
    win_size=16,
    train_stride=4,
    log_step=50,
    train_step=1,
    seed=2027,
    num_workers=4,
    eval_epoch_list=[i for i in range(6, 56)],
    use_filter=11,
    norm_type='real&image',
    aug_type='normal', # 'none' 'normal' 'mix'
    )

optim_cfg = dict(
    type='adamw',
    lr=0.00005,
)

schedule_cfg = dict(
    type='Cos',
    n_epoch=55,
    warmup_epoch=5,
)
