import os
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
from tqdm import tqdm
import numpy as np
import shutil
from cruw.eval.rod.rod_eval_utils import accumulate, summarize
from cruw.eval import evaluate_rodnet_seq
import matplotlib.pyplot as plt



def parse_args():
    parser = argparse.ArgumentParser(description='Test RODNet.')

    parser.add_argument('--config', type=str, default='/home/jackychou/code/RODNet_UAV/configs/rodnet-CDC-single-32_128_11_baseline_r&i.py', help='choose rodnet model configurations')
    parser.add_argument('--sensor_config', type=str, default='/home/jackychou/code/RODNet_UAV/cruw-devkit/cruw/dataset_configs/uniform_32.json')
    parser.add_argument('--data_dir', type=str, default='/home/jackychou/dataset/UAV1.0/train_test_32_11_128', help='directory to the prepared data')
    parser.add_argument('--checkpoint', type=str, default='/home/jackychou/code/RODNet_UAV/workers/1-PCA_Uniform/rodnet-CDC-single-32_128_11_baseline_r&i/rodnet-cdcv2-win16-mnet-20250720-053822/epoch_14_best.pkl', help='path to the saved trained model')
    parser.add_argument('--res_dir', type=str, default='/mnt/c/Ubuntu-temp/labels/pred_CDC', help='directory to save testing results')
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")
    parser.add_argument('--demo', action="store_true", help='False: test with GT, True: demo without GT')
    parser.add_argument('--symbol', action="store_true", help='use symbol or text+score')

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 参数初始化
    args = parse_args()
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)  # update configs by args
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid
    model_cfg = config_dict['model_cfg']
    n_class = dataset.object_cfg.n_class
    dataset_configs = config_dict['dataset_cfg']
    res_root = args.res_dir
    if not os.path.exists(res_root):
        os.mkdir(res_root)

    # 数据集加载
    eval_seq_names = dataset_configs['test']['seqs']
    eval_dataloader_list = {}
    for subset_idx in eval_seq_names:
        subset = f'uav_seqs_{subset_idx}'
        crdata_test = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict,
                                split='test', noise_channel=args.use_noise_channel, subset=subset,
                                is_random_chirp=False, use_training=False)
        eval_dataloader = DataLoader(crdata_test, batch_size=1, shuffle=False, num_workers=8, collate_fn=cr_collate, persistent_workers=True)
        eval_dataloader_list[subset] = eval_dataloader

    # 模型加载
    if model_cfg['type'] == 'CDCv2':
        from rodnet.models import RODNetCDCDCN as Model
    elif model_cfg['type'] == 'CDCv2STCT':
        from rodnet.models import RODNetCDCDCNSTCT as Model
    elif model_cfg['type'] == 'myNet':
        from rodnet.models import myNet as Model
    else:
        raise NotImplementedError

    dataset_configs = config_dict['dataset_cfg']
    train_configs = config_dict['train_cfg']
    test_configs = config_dict['test_cfg']

    win_size = train_configs['win_size']
    n_class = dataset.object_cfg.n_class
    confmap_shape = (n_class, radar_configs['ramap_rsize'] // 4, radar_configs['ramap_asize'])
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
    if model_cfg['type'] == 'CDCv2':
        in_chirps = len(radar_configs['chirp_ids'])
        model = Model(in_channels=in_chirps, n_class=n_class, mnet_cfg=config_dict['model_cfg']['mnet_cfg'], dcn=config_dict['model_cfg']['dcn']).cuda()
    elif model_cfg['type'] == 'CDCv2STCT':
        in_chirps = len(radar_configs['chirp_ids'])
        model = Model(in_channels=in_chirps, n_class=n_class, mnet_cfg=config_dict['model_cfg']['mnet_cfg']).cuda()
    elif 'myNet' == model_cfg['type']:
        model = Model(config_dict['model_cfg']['mnet_cfg'], n_class, config_dict['model_cfg']['mnet_type'], config_dict['model_cfg']['train_type'], config_dict['model_cfg']['head_size'], config_dict['model_cfg']['norm_type'], config_dict['model_cfg']['act_type'], config_dict['model_cfg']['full_conv']).cuda()
    else:
        raise TypeError

    checkpoint = torch.load(checkpoint_path)

    if 'optimizer_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    if 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
    else:
        model_name = create_random_model_name(model_cfg['name'], checkpoint_path)
    model.eval()

    model_dir = res_root
    # 推理
    n_frame = 300
    model.eval()
    with torch.no_grad():
        print("Start eval")
        save_result = False

        # Create Temp Dir
        test_res_dir = os.path.join(model_dir, "temp_results")
        if os.path.exists(test_res_dir):
            shutil.rmtree(test_res_dir)
        os.mkdir(test_res_dir)
        data_root = dataset_configs['data_root']
        test_root = os.path.join(data_root.split('uav')[0], args.data_dir, "val")
        if os.path.exists(test_root) is False:
            test_root = os.path.join(data_root.split('uav')[0], args.data_dir, "test")
        seq_names = sorted(os.listdir(test_root))
        seq_names = [file.replace('.pkl', '') for file in seq_names]
        print(seq_names)

        for seq_name in seq_names:
            seq_res_dir = os.path.join(test_res_dir, seq_name)
            if not os.path.exists(seq_res_dir):
                os.makedirs(seq_res_dir)
            seq_res_viz_dir = os.path.join(seq_res_dir, 'rod_viz')
            if not os.path.exists(seq_res_viz_dir):
                os.makedirs(seq_res_viz_dir)
            f = open(os.path.join(seq_res_dir, 'rod_res.txt'), 'w')
            f.close()

        for subset in tqdm(seq_names):
            eval_dataloader = eval_dataloader_list[subset]
            init_genConfmap = ConfmapStack(confmap_shape)
            iter_ = init_genConfmap
            for i in range(train_configs['win_size'] - 1):
                while iter_.next is not None:
                    iter_ = iter_.next
                iter_.next = ConfmapStack(confmap_shape)

            for iter, data_dict in enumerate(eval_dataloader):
                data = data_dict['radar_data']
                seq_name = data_dict['seq_names'][0]
                save_path = os.path.join(test_res_dir, seq_name, 'rod_res.txt')
                start_frame_id = data_dict['start_frame'].item()
                confmap_pred = model(data.float().cuda())

                if type(confmap_pred) is list:
                    confmap_pred = confmap_pred[0]

                confmap_pred = confmap_pred.sigmoid()
                confmap_pred = confmap_pred.cpu().detach().numpy()
                if args.use_noise_channel:
                    confmap_pred = confmap_pred[:, :n_class, :, :, :]

                # 保存confmap
                sub_save_root = os.path.join(model_dir, seq_name)
                if not os.path.exists(sub_save_root):
                    os.mkdir(sub_save_root)
                save_pred_source = np.squeeze(confmap_pred, axis=0)
                save_pred_source = np.squeeze(save_pred_source, axis=0)
                for i in range(save_pred_source.shape[0]):
                    sub_save_path = os.path.join(sub_save_root, f"{start_frame_id+i:09d}.jpg")
                    sub_save_pred = save_pred_source[i, ...]
                    plt.close('all')
                    fig = plt.figure()
                    plt.imshow(sub_save_pred, origin='lower')
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.savefig(sub_save_path, dpi=300, bbox_inches='tight', pad_inches=0)

                iter_ = init_genConfmap
                for i in range(confmap_pred.shape[2]):
                    if iter_.next is None and i != confmap_pred.shape[2] - 1:
                        iter_.next = ConfmapStack(confmap_shape)
                    iter_.append(confmap_pred[0, :, i, :, :])
                    iter_ = iter_.next

                for i in range(test_configs['test_stride']):
                    res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_dict)
                    cur_frame_id = start_frame_id + i
                    write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
                    init_genConfmap = init_genConfmap.next

                if iter == len(eval_dataloader) - 1:
                    offset = test_configs['test_stride']
                    cur_frame_id = start_frame_id + offset
                    while init_genConfmap is not None:
                        res_final = post_process_single_frame(init_genConfmap.confmap, dataset, config_dict)
                        write_dets_results_single_frame(res_final, cur_frame_id, save_path, dataset)
                        init_genConfmap = init_genConfmap.next
                        offset += 1
                        cur_frame_id += 1

                if init_genConfmap is None:
                    init_genConfmap = ConfmapStack(confmap_shape)

        olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
        recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
        dataset = CRUW(data_root=args.data_dir, sensor_config_name=args.sensor_config)
        seq_names = sorted(os.listdir(test_res_dir))
        seq_names = [name for name in seq_names if '.' not in name]
        evalImgs_all = []
        n_frames_all = 0

        # 可不需要
        # evalImgs_high = []
        # n_frames_high = 0
        # evalImgs_low = []
        # n_frames_low = 0
        # high_seq_names = ['uav_seqs_2', 'uav_seqs_37', 'uav_seqs_45', 'uav_seqs_47', 'uav_seqs_55', 'uav_seqs_59', 'uav_seqs_6', 'uav_seqs_63', 'uav_seqs_7', 'uav_seqs_71', 'uav_seqs_73', 'uav_seqs_74', 'uav_seqs_77', 'uav_seqs_8']
        # low_seq_names = ['uav_seqs_65', 'uav_seqs_71', 'uav_seqs_74', 'uav_seqs_77',]

        for seq_name in seq_names:
            seq_label_temp = f"annot/rodnet_labels_{str(radar_configs['ramap_rsize'] // dataset_configs['rangeDownSample'])}_rad.csv"
            gt_path = os.path.join(config_dict['dataset_cfg']['base_root'], seq_name, seq_label_temp)
            res_path = os.path.join(test_res_dir, seq_name, 'rod_res.txt')
            evalImgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, dataset)
            eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, dataset, log=False)
            stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
            print("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100, stats[2] * 100))

            n_frames_all += n_frame
            evalImgs_all.extend(evalImgs)

            # 可不需要
            # if seq_name in high_seq_names:
            #     n_frames_high += n_frame
            #     evalImgs_high.extend(evalImgs)
            # if seq_name in low_seq_names:
            #     n_frames_low += n_frame
            #     evalImgs_low.extend(evalImgs)

        eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, dataset, log=False)
        stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
        print("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f | mAR50:90: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100, stats[2] * 100, stats[3] * 100))

        # 可不需要
        # eval = accumulate(evalImgs_high, n_frames_high, olsThrs, recThrs, dataset, log=False)
        # stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
        # print("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f | mAR50:90: %.4f" % ('High'.ljust(18), stats[0] * 100, stats[1] * 100, stats[2] * 100, stats[3] * 100))
        # eval = accumulate(evalImgs_low, n_frames_low, olsThrs, recThrs, dataset, log=False)
        # stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
        # print("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f | mAR50:90: %.4f" % ('Low'.ljust(18), stats[0] * 100, stats[1] * 100, stats[2] * 100, stats[3] * 100))

        if not save_result:
            shutil.rmtree(test_res_dir)
    print(stats)




