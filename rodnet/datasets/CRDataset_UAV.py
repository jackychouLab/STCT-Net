import os
import time
import random
import pickle
import numpy as np
from torch.utils import data
from scipy.io import loadmat
from .loaders import list_pkl_filenames, list_pkl_filenames_from_prepared
import ast



class CRDataset(data.Dataset):
    def __init__(self, data_dir, dataset, config_dict, split, is_random_chirp=True, subset=None, noise_channel=False, use_training=False):
        # parameters settings

        self.data_dir = data_dir
        self.dataset = dataset
        self.config_dict = config_dict
        self.n_class = dataset.object_cfg.n_class
        self.win_size = config_dict['train_cfg']['win_size']
        self.split = split
        if split == 'train' or split == 'valid':
            self.step = config_dict['train_cfg']['train_step']
            self.stride = config_dict['train_cfg']['train_stride']
        else:
            self.step = config_dict['test_cfg']['test_step']
            self.stride = config_dict['test_cfg']['test_stride']
        self.is_random_chirp = is_random_chirp
        self.n_chirps = 1
        self.noise_channel = noise_channel
        self.norm_type = config_dict['train_cfg']['norm_type']
        self.mean_std_list = config_dict['dataset_cfg']['data_mean_std']
        self.rangeDownSample = config_dict['dataset_cfg']['rangeDownSample']
        self.use_filter = config_dict['train_cfg']['use_filter']
        self.crop_range = {}
        self.use_training = use_training
        self.print_num = 0
        # Dataloader for MNet
        if 'mnet_cfg' in self.config_dict['model_cfg']:
            in_chirps, out_channels = self.config_dict['model_cfg']['mnet_cfg']
            self.n_chirps = in_chirps
        self.chirp_ids = self.dataset.sensor_cfg.radar_cfg['chirp_ids']

        # dataset initialization
        self.image_paths = []
        self.radar_paths = []
        self.obj_infos = []
        self.confmaps = []
        self.n_data = 0
        self.index_mapping = []

        if subset is not None:
            self.data_files = [subset + '.pkl']
        else:
            self.data_files = list_pkl_filenames_from_prepared(data_dir, split)
        self.seq_names = [name.split('.')[0] for name in self.data_files]
        self.n_seq = len(self.seq_names)

        split_folder = split
        for seq_id, data_file in enumerate(self.data_files):
            data_file_path = os.path.join(data_dir, split_folder, data_file)
            data_details = pickle.load(open(data_file_path, 'rb'))
            if split == 'train' or split == 'valid':
                assert data_details['anno'] is not None
            n_frame = data_details['n_frame']
            self.image_paths.append(data_details['image_paths'])
            self.radar_paths.append(data_details['radar_paths'])
            n_data_in_seq = (n_frame - (self.win_size * self.step - 1)) // self.stride + (
                1 if (n_frame - (self.win_size * self.step - 1)) % self.stride > 0 else 0)
            self.n_data += n_data_in_seq
            for data_id in range(n_data_in_seq):
                self.index_mapping.append([seq_id, data_id * self.stride])
            if data_details['anno'] is not None:
                self.obj_infos.append(data_details['anno']['metadata'])
                self.confmaps.append(data_details['anno']['confmaps'])


    def __len__(self):
        """Total number of data/label pairs"""
        return self.n_data


    def read_mat(self, x):
        mat_data = loadmat(x)
        if 'RA_data' in mat_data:
            return np.array(mat_data['RA_data'])
        if 'RD_data' in mat_data:
            return np.array(mat_data['RD_data'])
        if 'AD_data' in mat_data:
            return np.array(mat_data['AD_data'])
        return None


    def __getitem__(self, index):

        seq_id, data_id = self.index_mapping[index]
        seq_name = self.seq_names[seq_id]
        image_paths = self.image_paths[seq_id]
        radar_paths = self.radar_paths[seq_id]
        if len(self.confmaps) != 0:
            this_seq_obj_info = self.obj_infos[seq_id]
            this_seq_confmap = self.confmaps[seq_id]

        data_dict = dict(
            status=True,
            seq_names=seq_name,
            image_paths=[]
        )

        if self.is_random_chirp:
            chirp_id = random.randint(0, len(self.chirp_ids) - 1)
        else:
            chirp_id = 0

        # Dataloader for MNet
        if 'mnet_cfg' in self.config_dict['model_cfg']:
            chirp_id = self.chirp_ids

        radar_configs = self.dataset.sensor_cfg.radar_cfg
        ramap_rsize = radar_configs['ramap_rsize']
        ramap_asize = radar_configs['ramap_asize']

        # Load radar data
        try:
            if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'AP':  # drop this format
                radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                for idx, frameid in enumerate(
                        range(data_id, data_id + self.win_size * self.step, self.step)):
                    radar_npy_win[idx, :, :, :] = np.load(radar_paths[frameid])
                    data_dict['image_paths'].append(image_paths[frameid])
            elif radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                if isinstance(chirp_id, int):
                    radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                    for idx, frameid in enumerate(
                            range(data_id, data_id + self.win_size * self.step, self.step)):

                        radar_npy_win[idx, :, :, :] = np.load(radar_paths[frameid][chirp_id])

                        data_dict['image_paths'].append(image_paths[frameid])
                elif isinstance(chirp_id, list):
                    radar_npy_win = np.zeros((self.win_size, self.n_chirps, ramap_rsize, ramap_asize, 2),
                                             dtype=np.float32)
                    for idx, frameid in enumerate(
                            range(data_id, data_id + self.win_size * self.step, self.step)):
                        for cid, c in enumerate(chirp_id):
                            npy_path = radar_paths[frameid][c]
                            radar_npy_win[idx, cid, :, :, :] = np.load(npy_path)
                        data_dict['image_paths'].append(image_paths[frameid])
                else:
                    raise TypeError
            elif radar_configs['data_type'] == 'ROD2021':
                if isinstance(chirp_id, int):
                    radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                    for idx, frameid in enumerate(range(data_id, data_id + self.win_size * self.step, self.step)):
                        if radar_paths[frameid][chirp_id][-3:] == 'npy':
                            radar_npy_win[idx, :, :, :] = np.load(radar_paths[frameid][chirp_id])
                        elif radar_paths[frameid][chirp_id][-3:] == 'mat':
                            radar_npy_win[idx, :, :, :] = self.read_mat(radar_paths[frameid][chirp_id])
                        else:
                            raise TypeError
                        if self.norm_type == "real&image":
                            if self.print_num <= 0:
                                print("NormType is real&image", self.use_filter)
                                self.print_num += 1
                            radar_npy_win[idx, :, :, 0] = (radar_npy_win[idx, :, :, 0] - self.mean_std_list[
                                f'RA{str(self.use_filter)}_real_mean']) / self.mean_std_list[
                                                              f'RA{str(self.use_filter)}_real_std']
                            radar_npy_win[idx, :, :, 1] = (radar_npy_win[idx, :, :, 1] - self.mean_std_list[
                                f'RA{str(self.use_filter)}_imag_mean']) / self.mean_std_list[
                                                              f'RA{str(self.use_filter)}_imag_std']
                        if self.norm_type == "mod&phase":
                            if self.print_num <= 0:
                                print("NormType is mod&phase", self.use_filter)
                                self.print_num += 1
                            raw_complex = radar_npy_win[idx, :, :, 0] + 1j * radar_npy_win[idx, :, :, 1]
                            radar_mod = np.abs(raw_complex)
                            radar_phase = np.angle(raw_complex)
                            radar_mod = (radar_mod - self.mean_std_list[f'RA{str(self.use_filter)}_mod_mean']) / self.mean_std_list[f'RA{str(self.use_filter)}_mod_std']
                            radar_phase = (radar_phase - self.mean_std_list[f'RA{str(self.use_filter)}_pha_mean']) / self.mean_std_list[f'RA{str(self.use_filter)}_pha_std']
                            norm_complex = radar_mod * np.exp(1j * radar_phase)
                            radar_npy_win[idx, :, :, 0] = np.real(norm_complex)
                            radar_npy_win[idx, :, :, 1] = np.imag(norm_complex)
                        if self.norm_type == "mod&phase&keep":
                            if self.print_num <= 0:
                                print("NormType is mod&phase&keep", self.use_filter)
                                self.print_num += 1
                            raw_complex = radar_npy_win[idx, :, :, 0] + 1j * radar_npy_win[idx, :, :, 1]
                            radar_mod = np.abs(raw_complex)
                            radar_phase = np.angle(raw_complex)
                            radar_mod = (radar_mod - self.mean_std_list[f'RA{str(self.use_filter)}_mod_mean']) / self.mean_std_list[f'RA{str(self.use_filter)}_mod_std']
                            radar_phase = (radar_phase - self.mean_std_list[f'RA{str(self.use_filter)}_pha_mean']) / self.mean_std_list[f'RA{str(self.use_filter)}_pha_std']
                            radar_npy_win[idx, :, :, 0] = radar_mod
                            radar_npy_win[idx, :, :, 1] = radar_phase
                        if self.print_num <= 0:
                            print("NormType is None")
                            self.print_num += 1
                        data_dict['image_paths'].append(image_paths[frameid])

                    crop_range_length = self.config_dict['radar_config']['ramap_rsize'] // self.rangeDownSample
                    adc_interval_path = os.path.join(self.config_dict['dataset_cfg']['base_root'], radar_paths[frameid][chirp_id].split('/')[-5], f"adc_interval/new_interval_{crop_range_length}.txt")
                    interval_seq_idx = str(radar_paths[frameid][chirp_id].split('/')[-5].split('_')[-1])
                    if interval_seq_idx not in self.crop_range:
                        with open(adc_interval_path, "r") as file:
                            content = file.read()
                            adc_interval = ast.literal_eval(content.split('\n')[-1])
                        self.crop_range[interval_seq_idx] = adc_interval
                    radar_npy_win = radar_npy_win[:, self.crop_range[interval_seq_idx][0]:self.crop_range[interval_seq_idx][1] + 1, ...]
                elif isinstance(chirp_id, list):
                    radar_npy_win = np.zeros((self.win_size, self.n_chirps, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                    for idx, frameid in enumerate(
                            range(data_id, data_id + self.win_size * self.step, self.step)):
                        for cid, c in enumerate(chirp_id):
                            npy_path = radar_paths[frameid][cid]
                            if npy_path[-3:] == 'npy':
                                radar_npy_win[idx, cid, :, :, :] = np.load(npy_path)
                            elif npy_path[-3:] == 'mat':
                                radar_npy_win[idx, cid, :, :, :] = self.read_mat(npy_path)
                            else:
                                raise TypeError
                            if self.norm_type == "real&image":
                                if self.print_num <= 0:
                                    print("NormType is real&image", self.use_filter)
                                    self.print_num += 1
                                radar_npy_win[idx, cid, :, :, 0] = (radar_npy_win[idx, cid, :, :, 0] - self.mean_std_list[
                                    f'RA{str(self.use_filter)}_real_mean']) / self.mean_std_list[
                                                                  f'RA{str(self.use_filter)}_real_std']
                                radar_npy_win[idx, cid, :, :, 1] = (radar_npy_win[idx, cid, :, :, 1] - self.mean_std_list[
                                    f'RA{str(self.use_filter)}_imag_mean']) / self.mean_std_list[
                                                                  f'RA{str(self.use_filter)}_imag_std']
                            if self.norm_type == "mod&phase":
                                if self.print_num <= 0:
                                    print("NormType is mod&phase", self.use_filter)
                                    self.print_num += 1
                                raw_complex = radar_npy_win[idx, cid, :, :, 0] + 1j * radar_npy_win[idx, cid, :, :, 1]
                                radar_mod = np.abs(raw_complex)
                                radar_phase = np.angle(raw_complex)
                                radar_mod = (radar_mod - self.mean_std_list[f'RA{str(self.use_filter)}_mod_mean']) / \
                                            self.mean_std_list[f'RA{str(self.use_filter)}_mod_std']
                                radar_phase = (radar_phase - self.mean_std_list[
                                    f'RA{str(self.use_filter)}_pha_mean']) / self.mean_std_list[
                                                  f'RA{str(self.use_filter)}_pha_std']
                                norm_complex = radar_mod * np.exp(1j * radar_phase)
                                radar_npy_win[idx, cid, :, :, 0] = np.real(norm_complex)
                                radar_npy_win[idx, cid, :, :, 1] = np.imag(norm_complex)
                            if self.norm_type == "mod&phase&keep":
                                if self.print_num <= 0:
                                    print("NormType is mod&phase&keep", self.use_filter)
                                    self.print_num += 1
                                raw_complex = radar_npy_win[idx, cid, :, :, 0] + 1j * radar_npy_win[idx, cid, :, :, 1]
                                radar_mod = np.abs(raw_complex)
                                radar_phase = np.angle(raw_complex)
                                radar_mod = (radar_mod - self.mean_std_list[f'RA{str(self.use_filter)}_mod_mean']) / \
                                            self.mean_std_list[f'RA{str(self.use_filter)}_mod_std']
                                radar_phase = (radar_phase - self.mean_std_list[
                                    f'RA{str(self.use_filter)}_pha_mean']) / self.mean_std_list[
                                                  f'RA{str(self.use_filter)}_pha_std']
                                radar_npy_win[idx, cid, :, :, 0] = radar_mod
                                radar_npy_win[idx, cid, :, :, 1] = radar_phase
                            if self.norm_type == "mod":
                                if self.print_num <= 0:
                                    print("NormType is mod", self.use_filter)
                                    self.print_num += 1
                                raw_complex = radar_npy_win[idx, cid, :, :, 0] + 1j * radar_npy_win[idx, cid, :, :, 1]
                                radar_mod = np.abs(raw_complex)
                                radar_phase = np.angle(raw_complex)
                                radar_mod = (radar_mod - self.mean_std_list[f'RA{str(self.use_filter)}_mod_mean']) / \
                                            self.mean_std_list[f'RA{str(self.use_filter)}_mod_std']
                                norm_complex = radar_mod * np.exp(1j * radar_phase)
                                radar_npy_win[idx, cid, :, :, 0] = radar_mod
                                radar_npy_win[idx, cid, :, :, 1] = norm_complex
                            if self.print_num <= 0:
                                print("NormType is None")
                                self.print_num += 1
                        data_dict['image_paths'].append(image_paths[frameid])
                    crop_range_length = self.config_dict['radar_config']['ramap_rsize'] // self.rangeDownSample
                    adc_interval_path = os.path.join(self.config_dict['dataset_cfg']['base_root'], radar_paths[idx][cid].split('/')[-5], f"adc_interval/new_interval_{str(crop_range_length)}.txt")
                    interval_seq_idx = str(radar_paths[idx][cid].split('/')[-5].split('_')[-1])
                    if interval_seq_idx not in self.crop_range:
                        with open(adc_interval_path, "r") as file:
                            content = file.read()
                            adc_interval = ast.literal_eval(content.split('\n')[-1])
                        self.crop_range[interval_seq_idx] = adc_interval
                    radar_npy_win = radar_npy_win[:, :, self.crop_range[interval_seq_idx][0]:self.crop_range[interval_seq_idx][1] + 1, ...]
                else:
                    raise TypeError
            else:
                raise NotImplementedError

            data_dict['start_frame'] = data_id
            data_dict['end_frame'] = data_id + self.win_size * self.step - 1

        except:
            # in case load npy fail
            data_dict['status'] = False
            if not os.path.exists('./tmp'):
                os.makedirs('./tmp')
            log_name = 'loadnpyfail-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
            with open(os.path.join('./tmp', log_name), 'w') as f_log:
                f_log.write('npy path: ' + radar_paths[frameid][chirp_id] + \
                            '\nframe indices: %d:%d:%d' % (data_id, data_id + self.win_size * self.step, self.step))
            return data_dict

        # Dataloader for MNet
        if 'mnet_cfg' in self.config_dict['model_cfg']:
            radar_npy_win = np.transpose(radar_npy_win, (4, 0, 1, 2, 3))
            assert radar_npy_win.shape == (2, self.win_size, self.n_chirps, radar_configs['ramap_rsize'] // self.rangeDownSample, radar_configs['ramap_asize'])
        else:
            radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))
            assert radar_npy_win.shape == (2, self.win_size, radar_configs['ramap_rsize'] // self.rangeDownSample, radar_configs['ramap_asize'])
        data_dict['radar_data'] = radar_npy_win
        # Load annotations
        if len(self.confmaps) != 0:
            confmap_gt = this_seq_confmap[data_id:data_id + self.win_size * self.step:self.step]
            confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
            obj_info = this_seq_obj_info[data_id:data_id + self.win_size * self.step:self.step]
            if self.noise_channel:
                assert confmap_gt.shape == \
                       (self.n_class + 1, self.win_size, radar_configs['ramap_rsize'] // self.rangeDownSample, radar_configs['ramap_asize'])
            else:
                confmap_gt = confmap_gt[:self.n_class]
                assert confmap_gt.shape == \
                       (self.n_class, self.win_size, radar_configs['ramap_rsize'] // self.rangeDownSample, radar_configs['ramap_asize'])

            data_dict['anno'] = dict(
                obj_infos=obj_info,
                confmaps=confmap_gt,
            )
        else:
            data_dict['anno'] = None

        if self.use_training:
            return data_dict['radar_data'], data_dict['anno']['confmaps'], data_dict['image_paths']
        return data_dict
