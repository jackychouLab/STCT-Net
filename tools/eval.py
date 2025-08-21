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



DataType = 'matlab' # 'matlab' 'python' '4dfft'
use_cluster_filter = 0
chirp_num = 16
label_type = "rad" # rad deg
config_name = f"test_{DataType}_{use_cluster_filter}_wo_n"

config_path = f"/home/jackychou/Zhou/Project/python_code/RODNet_UAV/configs/{config_name}.py"
sensor_config_path = f"/home/jackychou/Zhou/Project/python_code/RODNet_UAV/cruw-devkit/cruw/dataset_configs/{DataType}_config_UAV_{chirp_num}chirps.json"
gt_dir_path = "/home/jackychou/Zhou/Datasets/test_UAV_dataset"
data_dir_path = os.path.join(gt_dir_path, f"train_test_seqs_rodnet_python_{chirp_num}_{use_cluster_filter}")
work_dir_path = f"/home/jackychou/Zhou/Project/python_code/RODNet_UAV/workers/{config_name}"
sub_work_dir_name = os.listdir(work_dir_path)[0]
work_dir_path = os.path.join(work_dir_path, sub_work_dir_name)
save_result = False

checkpoint_path = os.path.join(work_dir_path, "epoch_100.pkl")
res_dir_path = os.path.join(work_dir_path, "results")

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
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)
    dataset = CRUW(data_root=config_dict['dataset_cfg']['base_root'], sensor_config_name=args.sensor_config)
    radar_configs = dataset.sensor_cfg.radar_cfg
    range_grid = dataset.range_grid
    angle_grid = dataset.angle_grid

    # eval setting
    dataset_configs = config_dict['dataset_cfg']
    train_configs = config_dict['train_cfg']
    test_configs = config_dict['test_cfg']
    win_size = train_configs['win_size']
    n_class = dataset.object_cfg.n_class
    confmap_shape = (n_class, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])



    rodnet.eval()

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
        seq_names = sorted(os.listdir(os.path.join(data_root.split('uav')[0], f'train_test_seqs_rodnet_{DataType}_{chirp_num}_{use_cluster_filter}', dataset_configs['test']['subdir'])))
    else:
        seq_names = sorted(os.listdir(os.path.join(data_root.split('uav')[0], f'train_test_seqs_rodnet_{DataType}_{chirp_num}_{use_cluster_filter}', dataset_configs['demo']['subdir'])))

    seq_names = [file.replace('.pkl', '') for file in seq_names]

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
        crdata_test = CRDataset(data_dir=args.data_dir, dataset=dataset, config_dict=config_dict, split='test', noise_channel=args.use_noise_channel, subset=subset, is_random_chirp=False, mean_std=config_dict['train_cfg']['mean_std'])

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
            confmap_pred = rodnet(data.float().cuda())

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
            load_tic = time.time()


    # eval
    olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
    recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)

    dataset = CRUW(data_root=args.data_dir, sensor_config_name=args.sensor_config)

    seq_names = sorted(os.listdir(args.res_dir))
    seq_names = [name for name in seq_names if '.' not in name]

    evalImgs_all = []
    n_frames_all = 0

    for seq_name in seq_names:
        gt_path = os.path.join(args.gt_dir, seq_name, 'annot/ramap_labels_rad.csv')
        res_path = os.path.join(args.res_dir, seq_name, 'rod_res.txt')
        n_frame = len(os.listdir(os.path.join(args.gt_dir, seq_name, dataset.sensor_cfg.camera_cfg['image_folder'])))
        evalImgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, dataset)
        eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, dataset, log=False)
        stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
        n_frames_all += n_frame
        evalImgs_all.extend(evalImgs)

    eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, dataset, log=False)
    stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
    print("%s | AP_total: %.4f | AR_total: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100))

    if not save_result:
        shutil.rmtree(args.res_dir)