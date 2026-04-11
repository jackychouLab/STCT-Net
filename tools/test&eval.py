import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from cruw import CRUW
from rodnet.datasets.CRDataset_UAV import CRDataset
from rodnet.datasets.collate_functions import cr_collate
from rodnet.core.post_processing import post_process_single_frame
from rodnet.core.post_processing import write_dets_results_single_frame
from rodnet.core.post_processing import ConfmapStack
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict
from rodnet.utils.solve_dir import create_random_model_name
import numpy as np
from cruw.eval.rod.rod_eval_utils import accumulate, summarize
from cruw.eval import evaluate_rodnet_seq
import shutil
from tqdm import tqdm


# revise start
chirp_type = 'pca' # pca or uniform
chirp_num = 32
use_cluster_filter = 1132
config_name = f"rodnet-myNet-single-32_128_1132_baseline_r&i"
workers_name = "2-optimizerBackbone/True_GN_GELU"
sub_work_dir_name = None
best_num = '23'
only_test = None
# SNR <  4.3: 6 45 55 59 65 71 73 74 77
# UNF:   48.42 69.74 50.99 57.46
# PCA:   64.24 81.53 69.57 72.22
# SNR >= 4.3: 2 7 8 37 47 63
# UNF:   30.72 54.89 32.08 40.66
# PCA:   74.60 97.32 82.80 85.33
# revise stop

range_length = 128
label_type = "rad" # rad deg
config_path = f"/home/jackychou/code/RODNet_UAV/configs/{config_name}.py"
sensor_config_path = f"/home/jackychou/code/RODNet_UAV/cruw-devkit/cruw/dataset_configs/{chirp_type}_{chirp_num}.json"
gt_dir_path = "/home/jackychou/dataset/UAV1.0"
data_dir_path = os.path.join(gt_dir_path, f"train_test_{chirp_num}_{use_cluster_filter}_{range_length}")
work_dir_path = f"/home/jackychou/code/RODNet_UAV/workers/{workers_name}"
if sub_work_dir_name is None:
    sub_work_dir_name = os.listdir(work_dir_path)[0]
work_dir_path = os.path.join(work_dir_path, sub_work_dir_name)
save_result = False

checkpoint_path = os.path.join(work_dir_path, f"epoch_{best_num}_best.pkl")
print(checkpoint_path)
print(os.path.exists(checkpoint_path))
print(os.path.getsize(checkpoint_path))
res_dir_path = os.path.join(work_dir_path, "test_results")

def parse_args():
    parser = argparse.ArgumentParser(description='Test RODNet.')

    parser.add_argument('--config', type=str, default=config_path, help='choose rodnet model configurations')
    parser.add_argument('--sensor_config', type=str, default=sensor_config_path)
    parser.add_argument('--data_dir', type=str, default=data_dir_path, help='directory to the prepared data')
    parser.add_argument('--checkpoint', type=str, default=checkpoint_path, help='path to the saved trained model')
    parser.add_argument('--res_dir', type=str, default=res_dir_path, help='directory to save testing results')
    parser.add_argument('--gt_dir', type=str, default=gt_dir_path, help='directory to save testing results')
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")
    parser.add_argument('--demo', action="store_true", help='False: test with GT, True: demo without GT')
    parser.add_argument('--symbol', action="store_true", help='use symbol or text+score')

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    sybl = args.symbol
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)  # update configs by args
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    model_cfg = config_dict['model_cfg']

    if model_cfg['type'] == 'CDC':
        from rodnet.models import RODNetCDC as Model
    elif model_cfg['type'] == 'HG':
        from rodnet.models import RODNetHG as Model
    elif model_cfg['type'] == 'HGwI':
        from rodnet.models import RODNetHGwI as Model
    elif model_cfg['type'] == 'CDCv2':
        from rodnet.models import RODNetCDCDCN as Model
    elif model_cfg['type'] == 'HGv2':
        from rodnet.models import RODNetHGDCN as Model
    elif model_cfg['type'] == 'HGwIv2':
        from rodnet.models import RODNetHGwIDCN as Model
    elif model_cfg['type'] == 'T':
        from rodnet.models import T_RODNet as Model
    elif model_cfg['type'] == 'E_RODNet':
        from rodnet.models import E_RODNet as Model
    elif model_cfg['type'] == 'DCSN':
        from rodnet.models import RODNet_DCSN as Model
    elif model_cfg['type'] == 'myNet':
        from rodnet.models import myNet as Model
    else:
        raise NotImplementedError

    # parameter settings
    dataset_configs = config_dict['dataset_cfg']
    train_configs = config_dict['train_cfg']
    test_configs = config_dict['test_cfg']

    win_size = train_configs['win_size']
    n_class = dataset.object_cfg.n_class
    confmap_shape = (n_class, radar_configs['ramap_rsize'] // (512 // range_length), radar_configs['ramap_asize'])
    if 'stacked_num' in model_cfg:
        stacked_num = model_cfg['stacked_num']
    else:
        stacked_num = None

    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        checkpoint_path = args.checkpoint
    else:
        raise ValueError("No trained model found.")

    if args.use_noise_channel:
        n_class_test = n_class + 1
    else:
        n_class_test = n_class

    print("Building model ... (%s)" % model_cfg)

    if model_cfg['type'] == 'CDC':
        model = Model(in_channels=2, n_class=n_class_test).cuda()
    elif model_cfg['type'] == 'HG':
        model = Model(in_channels=2, n_class=n_class_test, stacked_num=stacked_num).cuda()
    elif model_cfg['type'] == 'HGwI':
        model = Model(in_channels=2, n_class=n_class_test, stacked_num=stacked_num).cuda()
    elif model_cfg['type'] == 'CDCv2':
        in_chirps = len(radar_configs['chirp_ids'])
        model = Model(in_channels=in_chirps, n_class=n_class_test,
                           mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                           dcn=config_dict['model_cfg']['dcn']).cuda()
    elif model_cfg['type'] == 'HGv2':
        in_chirps = len(radar_configs['chirp_ids'])
        model = Model(in_channels=in_chirps, n_class=n_class_test, stacked_num=stacked_num,
                           mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                           dcn=config_dict['model_cfg']['dcn']).cuda()
    elif model_cfg['type'] == 'HGwIv2':
        in_chirps = len(radar_configs['chirp_ids'])
        model = Model(in_channels=in_chirps, n_class=n_class_test, stacked_num=stacked_num,
                           mnet_cfg=config_dict['model_cfg']['mnet_cfg'],
                           dcn=config_dict['model_cfg']['dcn']).cuda()
    elif model_cfg['type'] == 'T':
        model = Model(num_classes=n_class_test, embed_dim=64, win_size=4).cuda()
    elif model_cfg['type'] == 'E_RODNet':
        model = Model(config_dict['model_cfg']['mnet_cfg'], n_class_test).cuda()
    elif model_cfg['type'] == 'DCSN':
        model = Model(n_class_test, stacked_num=stacked_num).cuda()
    elif 'myNet' == model_cfg['type']:
        model = Model(config_dict['model_cfg']['mnet_cfg'], n_class_test, config_dict['model_cfg']['mnet_type'], config_dict['model_cfg']['train_type'], config_dict['model_cfg']['head_size'], config_dict['model_cfg']['norm_type'], config_dict['model_cfg']['act_type'], config_dict['model_cfg']['full_conv']).cuda()
    else:
        raise TypeError

    checkpoint = torch.load(checkpoint_path, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    else:
        model_name = create_random_model_name(model_cfg['name'], checkpoint_path)
    model.eval()



    test_res_dir = args.res_dir
    if os.path.exists(test_res_dir):
        shutil.rmtree(test_res_dir)
    os.mkdir(test_res_dir)

    # save current checkpoint path
    weight_log_path = os.path.join(test_res_dir, 'weight_name.txt')
    if os.path.exists(weight_log_path):
        with open(weight_log_path, 'a+') as f:
            f.write(checkpoint_path + '\n')
    else:
        with open(weight_log_path, 'w') as f:
            f.write(checkpoint_path + '\n')

    total_time = 0
    total_count = 0

    data_root = dataset_configs['data_root']

    if not args.demo:
        seq_names = sorted(os.listdir(os.path.join(data_root.split('uav')[0], f'train_test_{chirp_num}_{use_cluster_filter}_{range_length}', dataset_configs['test']['subdir'])))
    else:
        seq_names = sorted(os.listdir(os.path.join(data_root.split('uav')[0], f'train_test_{chirp_num}_{use_cluster_filter}_{range_length}', dataset_configs['demo']['subdir'])))

    seq_names = [file.replace('.pkl', '') for file in seq_names]
    if only_test is not None:
        new_seq_names = []
        for seq_name in seq_names:
            for sub_only_test in only_test:
                if f'uav_seqs_{sub_only_test}' == seq_name:
                    new_seq_names.append(seq_name)
                    break
        seq_names = new_seq_names

    for seq_name in seq_names:
        seq_res_dir = os.path.join(test_res_dir, seq_name)
        if not os.path.exists(seq_res_dir):
            os.makedirs(seq_res_dir)
        seq_res_viz_dir = os.path.join(seq_res_dir, 'rod_viz')
        if not os.path.exists(seq_res_viz_dir):
            os.makedirs(seq_res_viz_dir)
        f = open(os.path.join(seq_res_dir, 'rod_res.txt'), 'w')
        f.close()

    with torch.no_grad():
        for subset in tqdm(seq_names):
            crdata_test = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='test',
                      noise_channel=args.use_noise_channel, subset=subset, is_random_chirp=False, use_training=False)
            dataloader = DataLoader(crdata_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=cr_collate)

            seq_names = crdata_test.seq_names
            index_mapping = crdata_test.index_mapping

            init_genConfmap = ConfmapStack(confmap_shape)
            iter_ = init_genConfmap
            for i in range(train_configs['win_size'] - 1):
                while iter_.next is not None:
                    iter_ = iter_.next
                iter_.next = ConfmapStack(confmap_shape)

            load_tic = time.time()
            for iter, data_dict in enumerate(dataloader):
                load_time = time.time() - load_tic
                data = data_dict['radar_data']
                try:
                    image_paths = data_dict['image_paths'][0]
                except:
                    print('warning: fail to load RGB images, will not visualize results')
                    image_paths = None
                seq_name = data_dict['seq_names'][0]
                if not args.demo:
                    confmap_gt = data_dict['anno']['confmaps']
                    obj_info = data_dict['anno']['obj_infos']
                else:
                    confmap_gt = None
                    obj_info = None

                save_path = os.path.join(test_res_dir, seq_name, 'rod_res.txt')

                start_frame_id = data_dict['start_frame'].item()
                end_frame_id = data_dict['end_frame'].item()

                tic = time.time()
                confmap_pred = model(data.float().cuda())
                if stacked_num is not None:
                    confmap_pred = confmap_pred[-1].cpu().detach().numpy()  # (1, 4, 32, 128, 128)
                else:
                    confmap_pred = confmap_pred.cpu().detach().numpy()

                if args.use_noise_channel:
                    confmap_pred = confmap_pred[:, :n_class, :, :, :]

                infer_time = time.time() - tic
                total_time += infer_time

                iter_ = init_genConfmap
                for i in range(confmap_pred.shape[2]):
                    if iter_.next is None and i != confmap_pred.shape[2] - 1:
                        iter_.next = ConfmapStack(confmap_shape)
                    iter_.append(confmap_pred[0, :, i, :, :])
                    iter_ = iter_.next

                process_tic = time.time()
                for i in range(test_configs['test_stride']):
                    total_count += 1
                    res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_dict)
                    cur_frame_id = start_frame_id + i
                    write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
                    confmap_pred_0 = init_genConfmap.confmap
                    res_final_0 = res_final
                    init_genConfmap = init_genConfmap.next

                if iter == len(dataloader) - 1:
                    offset = test_configs['test_stride']
                    cur_frame_id = start_frame_id + offset
                    while init_genConfmap is not None:
                        total_count += 1
                        res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_dict)
                        write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
                        confmap_pred_0 = init_genConfmap.confmap
                        res_final_0 = res_final
                        init_genConfmap = init_genConfmap.next
                        offset += 1
                        cur_frame_id += 1

                if init_genConfmap is None:
                    init_genConfmap = ConfmapStack(confmap_shape)

                proc_time = time.time() - process_tic
                # print("Testing %s: frame %4d to %4d | Load time: %.4f | Inference time: %.4f | Process time: %.4f" %
                #       (seq_name, start_frame_id, end_frame_id, load_time, infer_time, proc_time))

                load_tic = time.time()

    print("ave time: %f" % (total_time / total_count))

    # eval
    olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
    recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)

    dataset = CRUW(data_root=args.data_dir, sensor_config_name=args.sensor_config)

    seq_names = sorted(os.listdir(args.res_dir))
    seq_names = [name for name in seq_names if '.' not in name]
    if only_test is not None:
        new_seq_names = []
        for seq_name in seq_names:
            for sub_only_test in only_test:
                if f'uav_seqs_{sub_only_test}' == seq_name:
                    new_seq_names.append(seq_name)
                    break
        seq_names = new_seq_names

    evalImgs_all = []
    n_frames_all = 0

    for seq_name in seq_names:
        gt_path = os.path.join(args.gt_dir, seq_name, f'annot/rodnet_labels_{range_length}_rad_{use_cluster_filter}.csv')
        res_path = os.path.join(args.res_dir, seq_name, 'rod_res.txt')
        n_frame = len(os.listdir(os.path.join(args.gt_dir, seq_name, dataset.sensor_cfg.camera_cfg['image_folder'])))
        evalImgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, dataset)
        eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, dataset, log=False)
        stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
        print("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100, stats[2] * 100))

        n_frames_all += n_frame
        evalImgs_all.extend(evalImgs)

    eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, dataset, log=False)
    stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
    print_gas = 0.4 * stats[0] + 0.4 * stats[1] + 0.2 * stats[2]
    print("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f | GAS: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100, stats[2] * 100, print_gas * 100))
    print(stats)
    if not save_result:
        shutil.rmtree(test_res_dir)