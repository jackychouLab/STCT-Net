import os
import random
import time
import json
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, SequentialLR, LinearLR, CyclicLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from cruw import CRUW
from rodnet.datasets.CRDataset_RAMP import CRDataset
from rodnet.datasets.collate_functions import cr_collate
from rodnet.core.radar_processing import chirp_amp
from rodnet.utils.solve_dir import create_dir_for_new_model
from rodnet.utils.load_configs import load_configs_from_file, parse_cfgs, update_config_dict
from rodnet.utils.visualization import visualize_train_img
from cruw.eval.rod.rod_eval_utils import accumulate, summarize
from cruw.eval import evaluate_rodnet_seq
import shutil
from rodnet.core.post_processing import ConfmapStack
from rodnet.core.post_processing import post_process_single_frame
from rodnet.core.post_processing import write_dets_results_single_frame
from tqdm import tqdm
import torch
import warnings
from data_aug import Aug_data
import math
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def _neg_loss(pred, gt):
    pred = torch.clamp(pred, 1.4013e-45, 1)
    fos = torch.sum(gt, 1)
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    fos_inds = torch.unsqueeze(fos.gt(0).float(), 1)
    fos_inds = fos_inds.expand(-1, 3, -1, -1, -1)
    neg_inds = neg_inds + (4 - 1) * fos_inds * gt.eq(0)

    neg_weights = torch.pow(1 - gt, 4)
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds * 4
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _MSE_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    num_pos = pos_inds.float().sum()
    mse_loss = torch.pow(pred - gt, 2)
    mse_loss = mse_loss.sum()

    if num_pos > 0:
        mse_loss = mse_loss / num_pos

    return mse_loss


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class Cont_Loss(nn.Module):
    def __init__(self):
        super(Cont_Loss, self).__init__()
        self.MSE_loss = nn.MSELoss()

    def forward(self, out, target):
        loss = 0
        for i in range(out.shape[0]):
            batch_loss = 0
            for j in range(16):
                if j % 2 == 0:
                    batch_loss = batch_loss + self.MSE_loss(out[i, :, j, :, :], target[i, :, j + 1, :, :])
            loss = loss + batch_loss/(16/2)
        return loss


class Trainer:
    def __init__(self, config_dict, args):
        self.config_dict = config_dict
        self.args = args
        self.dataset = CRUW(data_root=self.config_dict['dataset_cfg']['base_root'], sensor_config_name=self.args.sensor_config)
        self.radar_configs = self.dataset.sensor_cfg.radar_cfg
        self.optim_configs = self.config_dict['optim_cfg']
        self.schedule_configs = self.config_dict['schedule_cfg']
        self.range_grid = self.dataset.range_grid
        self.angle_grid = self.dataset.angle_grid
        self.dataset_configs = self.config_dict['dataset_cfg']
        self.train_configs = self.config_dict['train_cfg']
        self.test_configs = self.config_dict['test_cfg']
        self.win_size = self.train_configs['win_size']
        self.n_class = self.dataset.object_cfg.n_class
        self.confmap_shape = (self.n_class, self.radar_configs['ramap_rsize'] // self.dataset_configs['rangeDownSample'], self.radar_configs['ramap_asize'])
        self.model_cfg = self.config_dict['model_cfg']

        if not os.path.exists(self.args.log_dir):
            os.makedirs(self.args.log_dir)

        self.train_model_path = self.args.log_dir
        self.model_dir, self.model_name = create_dir_for_new_model(self.model_cfg['name'], self.train_model_path)

        self.train_viz_path = os.path.join(self.model_dir, 'train_viz')
        if not os.path.exists(self.train_viz_path):
            os.makedirs(self.train_viz_path)

        self.writer = SummaryWriter(self.model_dir)
        save_config_dict = {
            'args': vars(self.args),
            'config_dict': self.config_dict,
        }
        config_json_name = os.path.join(self.model_dir, 'config-' + time.strftime("%Y%m%d-%H%M%S") + '.json')
        with open(config_json_name, 'w') as fp:
            json.dumps(save_config_dict, cls=NpEncoder)

        self.train_log_name = os.path.join(self.model_dir, "train.log")
        with open(self.train_log_name, 'w'):
            pass

        self.n_class = self.dataset.object_cfg.n_class
        self.n_epoch = self.schedule_configs['n_epoch']
        self.batch_size = self.train_configs['batch_size']

        if 'stacked_num' in self.model_cfg:
            self.stacked_num = self.model_cfg['stacked_num']
        else:
            self.stacked_num = None

        self.crdata_train = CRDataset(data_dir=args.data_dir, dataset=self.dataset, config_dict=self.config_dict, split='train', noise_channel=self.args.use_noise_channel, use_training=True)
        self.seq_names = self.crdata_train.seq_names
        self.index_mapping = self.crdata_train.index_mapping
        self.train_dataloader = DataLoader(self.crdata_train, self.batch_size, shuffle=True, num_workers=self.train_configs['num_workers'])

        self.eval_seq_names = self.dataset_configs['test']['seqs']
        self.eval_dataloader_list = {}
        for subset_idx in self.eval_seq_names:
            subset = f'uav_seqs_{subset_idx}'
            crdata_test = CRDataset(data_dir=self.args.data_dir, dataset=self.dataset, config_dict=self.config_dict, split='test', noise_channel=self.args.use_noise_channel, subset=subset, is_random_chirp=False, use_training=False)
            eval_dataloader = DataLoader(crdata_test, batch_size=1, shuffle=False, num_workers=1, collate_fn=cr_collate, multiprocessing_context='spawn')
            self.eval_dataloader_list[subset] = eval_dataloader

        if self.args.use_noise_channel:
            n_class_train = self.n_class + 1
        else:
            n_class_train = self.n_class

        if self.model_cfg['type'] == 'RAMP':
            from rodnet.models import RAMP as Model
        else:
            raise NotImplementedError

        print("Building model ... (%s)" % self.model_cfg)
        if self.model_cfg['type'] == 'RAMP':
            self.model = Model(1, 16).cuda()
        else:
            raise TypeError

        if self.optim_configs['type'] == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.optim_configs['lr'])

        iter_num = len(self.train_dataloader)
        if self.schedule_configs['type'] == 'MultiStepLR':
            self.scheduler = SequentialLR(
                optimizer=self.optimizer,
                schedulers=[LinearLR(self.optimizer, start_factor=1e-7 / self.optim_configs['lr'],
                                     total_iters=self.schedule_configs['warmup_iters']),
                            MultiStepLR(self.optimizer,
                                        milestones=[temp_num * iter_num - self.schedule_configs['warmup_iters'] for temp_num in self.schedule_configs['milestones']],
                                        gamma=self.schedule_configs['gama'])],
                milestones=[self.schedule_configs['warmup_iters']]
            )

        if self.schedule_configs['type'] == 'CosStepLR':
            self.scheduler = SequentialLR(
                optimizer=self.optimizer,
                schedulers=[CyclicLR(self.optimizer, base_lr=self.optim_configs['lr'], max_lr=self.optim_configs['lr'] * 10, step_size_up=iter_num // 2, step_size_down=iter_num // 2),
                            MultiStepLR(self.optimizer, milestones=[temp_num * iter_num - (iter_num * 10) for temp_num in self.schedule_configs['milestones']], gamma=self.schedule_configs['gama'])],
                milestones=[iter_num * 10]
            )

        if self.schedule_configs['type'] == 'Cos':
            self.scheduler = SequentialLR(
                optimizer=self.optimizer,
                schedulers=[LinearLR(self.optimizer, start_factor=1e-7 / self.optim_configs['lr'],
                                     total_iters=self.schedule_configs['warmup_epoch'] * iter_num),
                            CosineAnnealingLR(self.optimizer, int(self.n_epoch * iter_num - self.schedule_configs['warmup_epoch'] * iter_num), eta_min=1e-6)],
                milestones=[self.schedule_configs['warmup_epoch'] * iter_num]
            )

        self.criterion = FocalLoss()
        self.criterion_2 = Cont_Loss()

        # print training configurations
        print("Model name: %s" % self.model_name)
        print("Number of sequences to train: %d" % self.crdata_train.n_seq)
        print("Training dataset length: %d" % len(self.crdata_train))
        print("Batch size: %d" % self.batch_size)
        print("Number of iterations in each epoch: %d" % int(len(self.crdata_train) / self.batch_size))

        self.iter_count = 0
        self.loss_ave = 0
        self.best_map = 0.
        self.best_ap50 = 0.
        self.best_ap70 = 0.
        self.best_score = 0.
        self.best_score_epoch = -1
        self.all_iters = len(self.train_dataloader) * self.n_epoch
        self.patience_count = 0


    def train(self, ):
        for epoch in range(self.n_epoch):
            self.model.train()
            tic_load = time.time()
            for iter, data_dict in enumerate(self.train_dataloader):
                data_ra, data_rv, data_va = data_dict[0]
                confmap_gt = data_dict[1]
                image_paths = data_dict[2]
                if self.config_dict['train_cfg']['aug_type'] != 'none':
                    data_ra, data_rv, data_va, confmap_gt = Aug_data(data_ra, data_rv, data_va, confmap_gt, type=self.config_dict['train_cfg']['aug_type'])
                tic = time.time()
                self.optimizer.zero_grad()
                confmap_preds, confmap_preds2 = self.model(data_ra.float().cuda(), data_rv.float().cuda(), data_va.float().cuda())
                loss_confmap = 0
                if self.stacked_num is not None:
                    for i in range(self.stacked_num):
                        loss_cur = self.criterion(confmap_preds[i], confmap_gt.float().cuda())
                        loss_confmap += loss_cur
                else:
                    loss_base_1 = self.criterion(confmap_preds, confmap_gt.float().cuda())
                    loss_base_2 = self.criterion_2(confmap_preds, confmap_gt.float().cuda())
                    loss_exo = self.criterion(confmap_preds2, confmap_gt.float().cuda())
                    loss_confmap += loss_base_1 + loss_base_2 + loss_exo * 0.5
                if math.isnan(loss_confmap.item()):
                    break
                loss_confmap.backward()
                self.optimizer.step()
                self.scheduler.step()
                tic_back = time.time()
                self.loss_ave = np.average([self.loss_ave, loss_confmap.item()], weights=[self.iter_count, 1])
                self.all_iters -= 1
                if iter % self.config_dict['train_cfg']['log_step'] == 0:
                    # print statistics
                    load_time = tic - tic_load
                    back_time = tic_back - tic
                    wait_time = (load_time + back_time) * self.all_iters
                    wait_h, wait_r = divmod(wait_time, 3600)
                    wait_m, wait_s = divmod(wait_r, 60)
                    print('epoch %2d, iter %4d: loss: %.4f (%.4f) | load time: %.4f | back time: %.4f | lr: %.8f | best AP50 acc: %.4f | best AP70 acc: %.4f | best mAP acc: %.4f | best Score: %.4f | best Score epoch: %2d | wait time: %2d : %2d : %2d' % (epoch + 1, iter + 1, loss_confmap.item(), self.loss_ave, load_time, back_time, self.optimizer.param_groups[0]['lr'], self.best_ap50, self.best_ap70, self.best_map, self.best_score, self.best_score_epoch, wait_h, wait_m, wait_s))
                    with open(self.train_log_name, 'a+') as f_log:
                        f_log.write('epoch %2d, iter %4d: loss: %.4f (%.4f) | load time: %.4f | back time: %.4f | lr: %.8f | best AP50 acc: %.4f | best AP70 acc: %.4f | best mAP acc: %.4f | best Score: %.4f | best Score epoch: %2d \n' % (epoch + 1, iter + 1, loss_confmap.item(), self.loss_ave, load_time, back_time, self.optimizer.param_groups[0]['lr'], self.best_ap50, self.best_ap70, self.best_map, self.best_score, self.best_score_epoch))

                    self.writer.add_scalar('loss/loss_all', loss_confmap.item(), self.iter_count)
                    self.writer.add_scalar('loss/loss_ave', self.loss_ave, self.iter_count)
                    self.writer.add_scalar('time/time_load', load_time, self.iter_count)
                    self.writer.add_scalar('time/time_back', back_time, self.iter_count)
                    self.writer.add_scalar('param/param_lr', self.scheduler.get_last_lr()[0], self.iter_count)

                    if self.stacked_num is not None:
                        confmap_pred = confmap_preds[self.stacked_num - 1]
                    else:
                        if type(confmap_preds) is list:
                            confmap_pred = confmap_preds[-1]
                        else:
                            confmap_pred = confmap_preds
                    confmap_pred = confmap_pred.cpu().detach().numpy()

                    if 'mnet_cfg' in self.model_cfg:
                        chirp_amp_curr = chirp_amp(data_ra.numpy()[0, :, 0, 0, :, :], self.radar_configs['data_type'])
                    else:
                        chirp_amp_curr = chirp_amp(data_ra.numpy()[0, :, 0, :, :], self.radar_configs['data_type'])

                    fig_name = os.path.join(self.train_viz_path, '%03d_%010d_%06d.png' % (epoch + 1, self.iter_count, iter + 1))
                    img_path = image_paths[0][0]
                    visualize_train_img(fig_name, img_path, chirp_amp_curr, confmap_pred[0, :self.n_class, 0, :, :], confmap_gt[0, :self.n_class, 0, :, :])
                self.iter_count += 1
                tic_load = time.time()

            if epoch + 1 in self.config_dict['train_cfg']['eval_epoch_list']:
                self.model.eval()
                map, ap50, ap70 = self.eval()
                score = map * 0.4 + ap50 * 0.4 + ap70 * 0.2
                self.model.train()

                if score > self.best_score:
                    self.patience_count = 0
                    self.best_map = map
                    self.best_ap50 = ap50
                    self.best_ap70 = ap70
                    self.best_score = score
                    self.best_score_epoch = epoch + 1
                    print("saving current model ...")
                    status_dict = {
                        'model_name': self.model_name,
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }
                    save_model_path = '%s/epoch_%02d_best.pkl' % (self.model_dir, epoch + 1)
                    torch.save(status_dict, save_model_path)
                else:
                    self.patience_count += 1

            if epoch + 1 == self.n_epoch or self.patience_count == 10:
                status_dict = {
                    'model_name': self.model_name,
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
                save_model_path = '%s/epoch_%02d_final.pkl' % (self.model_dir, epoch + 1)
                torch.save(status_dict, save_model_path)
                print('Training Finished.')
                break

            if math.isnan(loss_confmap.item()):
                print('Loss is NaN.')
                print('Training Finished.')
                break

    def eval(self):
        n_frame = 300
        self.model.eval()
        with torch.no_grad():
            print("Start eval")
            save_result = False

            # Create Temp Dir
            test_res_dir = os.path.join(self.model_dir, "temp_results")
            if os.path.exists(test_res_dir):
                shutil.rmtree(test_res_dir)
            os.mkdir(test_res_dir)
            data_root = self.dataset_configs['data_root']
            test_root = os.path.join(data_root.split('uav')[0], self.args.data_dir, "val")
            if os.path.exists(test_root) is False:
                test_root = os.path.join(data_root.split('uav')[0], self.args.data_dir, "test")
            seq_names = sorted(os.listdir(test_root))
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
                eval_dataloader = self.eval_dataloader_list[subset]
                init_genConfmap = ConfmapStack(self.confmap_shape)
                iter_ = init_genConfmap
                for i in range(self.train_configs['win_size'] - 1):
                    while iter_.next is not None:
                        iter_ = iter_.next
                    iter_.next = ConfmapStack(self.confmap_shape)

                for iter, data_dict in enumerate(eval_dataloader):
                    data_ra, data_rv, data_va = data_dict['ra'], data_dict['rd'], data_dict['ad']
                    seq_name = data_dict['seq_names'][0]
                    save_path = os.path.join(test_res_dir, seq_name, 'rod_res.txt')
                    start_frame_id = data_dict['start_frame'].item()
                    confmap_pred, _ = self.model(data_ra.float().cuda(), data_rv.float().cuda(), data_va.float().cuda())

                    if type(confmap_pred) is list:
                        confmap_pred = confmap_pred[0]

                    confmap_pred = confmap_pred.cpu().detach().numpy()
                    if args.use_noise_channel:
                        confmap_pred = confmap_pred[:, :self.n_class, :, :, :]

                    iter_ = init_genConfmap
                    for i in range(confmap_pred.shape[2]):
                        if iter_.next is None and i != confmap_pred.shape[2] - 1:
                            iter_.next = ConfmapStack(self.confmap_shape)
                        iter_.append(confmap_pred[0, :, i, :, :])
                        iter_ = iter_.next

                    for i in range(self.test_configs['test_stride']):
                        res_final = post_process_single_frame(init_genConfmap.confmap, self.dataset, self.config_dict)
                        cur_frame_id = start_frame_id + i
                        write_dets_results_single_frame(res_final, cur_frame_id, save_path, self.dataset)
                        init_genConfmap = init_genConfmap.next

                    if iter == len(eval_dataloader) - 1:
                        offset = self.test_configs['test_stride']
                        cur_frame_id = start_frame_id + offset
                        while init_genConfmap is not None:
                            res_final = post_process_single_frame(init_genConfmap.confmap, self.dataset, self.config_dict)
                            write_dets_results_single_frame(res_final, cur_frame_id, save_path, self.dataset)
                            init_genConfmap = init_genConfmap.next
                            offset += 1
                            cur_frame_id += 1

                    if init_genConfmap is None:
                        init_genConfmap = ConfmapStack(self.confmap_shape)

            olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
            recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
            dataset = CRUW(data_root=args.data_dir, sensor_config_name=args.sensor_config)
            seq_names = sorted(os.listdir(test_res_dir))
            seq_names = [name for name in seq_names if '.' not in name]
            evalImgs_all = []
            n_frames_all = 0

            for seq_name in seq_names:
                seq_label_temp = f"annot/rodnet_labels_{str(self.radar_configs['ramap_rsize'] // self.dataset_configs['rangeDownSample'])}_rad_{str(self.train_configs['use_filter'])}.csv"
                gt_path = os.path.join(config_dict['dataset_cfg']['base_root'], seq_name, seq_label_temp)
                res_path = os.path.join(test_res_dir, seq_name, 'rod_res.txt')
                evalImgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, dataset)
                eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, dataset, log=False)
                stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
                print("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100, stats[2] * 100))
                with open(self.train_log_name, 'a+') as f_log:
                    f_log.write("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f \n" % (seq_name.upper(), stats[0] * 100, stats[1] * 100, stats[2] * 100))

                n_frames_all += n_frame
                evalImgs_all.extend(evalImgs)

            eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, dataset, log=False)
            stats = summarize(eval, olsThrs, recThrs, dataset, gl=False)
            print("%s | mAP50:90: %.4f | AP50: %.4f | AP70: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100, stats[2] * 100))
            with open(self.train_log_name, 'a+') as f_log:
                f_log.write("%s | mAP50:95: %.4f | AP50: %.4f | AP70: %.4f \n" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100, stats[2] * 100))
            if not save_result:
                shutil.rmtree(test_res_dir)
        self.model.train()
        print(stats)
        return stats[0], stats[1], stats[2]


def parse_args():
    parser = argparse.ArgumentParser(description='Train RODNet.')
    parser.add_argument('--config', type=str, help='configuration file path')
    parser.add_argument('--sensor_config', type=str, help='sensor configuration file path')
    parser.add_argument('--data_dir', type=str, help='directory to the prepared data')
    parser.add_argument('--log_dir', type=str, help='directory to save trained model')
    parser.add_argument('--use_noise_channel', action="store_true", help="use noise channel or not")

    parser = parse_cfgs(parser)
    args = parser.parse_args()
    return args


def set_seed(seed, use_benchmark=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if use_benchmark:
        torch.backends.cudnn.benchmark = False
    print(f"Seed is setting to {seed}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    config_dict = load_configs_from_file(args.config)
    config_dict = update_config_dict(config_dict, args)
    set_seed(config_dict['train_cfg']['seed'])
    trainer = Trainer(config_dict, args)
    trainer.train()
