import os
import numpy as np
from scipy.signal.windows import taylor
from tqdm import tqdm
from scipy.signal import butter, lfilter
from concurrent.futures import ThreadPoolExecutor, as_completed



def butter_lowpass(order=5):
    b, a = butter(order, 0.707, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, order=5):
    b, a = butter_lowpass(order=order)
    y = lfilter(b, a, data)
    return y


def merge_creat_new(x):
    new = [item for sublist in x for item in sublist]
    out = []
    for i in new:
        if i not in out:
            out.append(i)
    return out



# 定义雷达对象参数
class RadarObject():
    def __init__(self):
        # 路径配置
        self.root = "/home/jackychou/Zhou/dataset/UAVRadar"
        self.save_root = "/home/jackychou/Zhou/dataset/UAVRadar"
        self.start_single_id = -1
        self.end_single_id = -1
        self.idx_list = [i for i in range(1, 79 + 1)]

        self.create_type = ['Azimuth']  # 'Azimuth', 'Elevation'

        # 6843 -- 配置
        self.numTX = 3
        self.numRX = 4
        self.numADCSamples = 512
        self.idxProcChirp = 255
        self.numLanes = 2
        self.framePerSecond = 10
        self.duration = 30

        # FFT -- 配置
        self.rangeBins = 512
        self.dopplerBins = 128
        self.azimuthBins = 128
        self.crop_num = 12
        self.rangeBins = self.rangeBins + 2 * self.crop_num
        self.use_filter = use_filter  # 仍然使用外部全局变量

        # 定义文件保存列表
        self.radarDataFileNameGroup = []
        self.saveDirNameGroup = []
        self.rgbFileNameGroup = []
        self.jointsFileNameGroup = []
        self.initialize()

        chirp_list = [
            [i for i in range(1, 256)],
            # [1, 85, 170, 255],
            # [1, 37, 73, 109, 146, 182, 218, 255],
            # [1, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255],
            # [1, 9, 17, 25, 33, 41, 50, 58, 66, 74, 82, 91, 99, 107, 115, 123, 132, 140, 148, 156, 164, 173, 181, 189, 197, 205, 214, 222, 230, 238, 246, 255],
            # [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 130, 134, 138, 142, 146, 150, 154, 158, 162, 166, 170, 174, 178, 182, 186, 190, 194, 198, 202, 206, 210, 214, 218, 222, 226, 230, 234, 238, 242, 246, 250, 255, 128],
        ]

        self.chirp_list = merge_creat_new(chirp_list)

        assert self.use_filter in [0, 1, 2, 3, 4, 10, 11, 22, 33, 44]

    def initialize(self):
        for i in self.idx_list:
            radarDataFileName = [
                os.path.join(self.root, f"uav_seqs_{i}", "raw_radar", "azimuth"),
                os.path.join(self.root, f"uav_seqs_{i}", "raw_radar", "elevation")
            ]
            seq_path = os.path.join(self.save_root, f"uav_seqs_{i}")
            saveDirName = os.path.join(seq_path, f"python_slice_frame_{str(self.use_filter)}")
            self.check_path([seq_path, saveDirName])
            self.radarDataFileNameGroup.append(radarDataFileName)
            self.saveDirNameGroup.append(saveDirName)

    def check_same(self, x, y):
        x = np.array(x).reshape(-1)
        y = np.array(y).reshape(-1)
        return np.array_equal(x, y)

    def check_path(self, x):
        # === 改成多线程安全版本 ===
        if isinstance(x, str):
            x = [x]
        for x_i in x:
            os.makedirs(x_i, exist_ok=True)

    def getAdcDataFromDCA1000(self, fileName):
        adcDataName = sorted(os.listdir(fileName))
        adcData = []
        for i, adc_data in enumerate(adcDataName):
            adc_data_tmp = np.fromfile(os.path.join(fileName, adcDataName[i]), dtype=np.int16)
            adcData.append(adc_data_tmp)
        adcData = np.concatenate(adcData)
        adcData = np.transpose(adcData.reshape(-1, 2, 2), axes=(0, 2, 1)).reshape(-1, 2)
        adcData = adcData[:, 0] + (np.sqrt(-1 + 0j) * adcData[:, 1])
        adcData = np.transpose(adcData.reshape(-1, self.numRX, self.numADCSamples), axes=(1, 0, 2))
        return adcData

    def clutterRemoval(self, input_val, axis=0):
        reordering = np.arange(len(input_val.shape))
        reordering[0] = axis
        reordering[axis] = 0
        input_val = input_val.transpose(reordering)

        # Apply static clutter removal
        mean = input_val.transpose(reordering).mean(0)
        output_val = input_val - np.expand_dims(mean, axis=0)
        out = output_val.transpose(reordering)
        return out

    def generateHeatmap(self, frame):
        data_first = frame[:, 0::3, :]
        data_second = frame[:, 1::3, :]
        data_third = frame[:, 2::3, :]
        # Azimuth-Doppler-Range
        data_merge = np.concatenate([data_first, data_second, data_third], axis=0)
        if self.use_filter in [2, 4, 22, 44]:
            for temp_idx in range(data_merge.shape[0]):
                for temp_jdx in range(data_merge.shape[1]):
                    data_merge[temp_idx, temp_jdx, :] = butter_lowpass_filter(
                        data_merge[temp_idx, temp_jdx, :], 6
                    )

        if self.use_filter in [1, 4, 11, 44]:
            data_merge = data_merge - np.mean(data_merge, axis=1, keepdims=True)

        # Range-FFT
        range_win = np.hamming(self.numADCSamples)
        merge_range = np.fft.fft(
            data_merge * range_win[np.newaxis, np.newaxis, :],
            axis=2,
            n=self.rangeBins
        )

        # create RD
        merge_RD = np.fft.fft(merge_range, axis=1, n=self.dopplerBins)
        merge_RD = np.fft.fftshift(merge_RD, axes=1)
        RD = np.mean(
            merge_RD[:, :, self.crop_num:-self.crop_num],
            axis=0,
            keepdims=False
        )
        RD = np.transpose(RD, axes=[1, 0])
        RD = np.concatenate(
            [np.expand_dims(np.real(RD), axis=-1),
             np.expand_dims(np.imag(RD), axis=-1)],
            axis=-1
        )

        # create RA
        azimuth_win = taylor(self.numRX * self.numTX)
        if self.use_filter in [11, 22, 44, 10]:
            azimuth_win = taylor(self.numRX * 2)
            RA = np.fft.fft(
                np.concatenate([merge_range[0:4, ...], merge_range[-4:, ...]], axis=0)
                * azimuth_win[:, np.newaxis, np.newaxis],
                axis=0,
                n=self.azimuthBins
            )
        else:
            RA = np.fft.fft(
                merge_range * azimuth_win[:, np.newaxis, np.newaxis],
                axis=0,
                n=self.azimuthBins
            )
        RA = np.fft.fftshift(RA, axes=0)
        RA = RA[:, :, self.crop_num:-self.crop_num]
        RA = np.transpose(RA, axes=[1, 2, 0])
        RA = np.concatenate(
            [np.expand_dims(np.real(RA), axis=-1),
             np.expand_dims(np.imag(RA), axis=-1)],
            axis=-1
        )

        # create AD
        if self.use_filter in [11, 22, 33, 44, 10]:
            azimuth_win = taylor(self.numRX * 2)
            AD = np.fft.fft(
                np.concatenate([merge_RD[0:4, ...], merge_RD[-4:, ...]], axis=0)
                * azimuth_win[:, np.newaxis, np.newaxis],
                axis=0,
                n=self.azimuthBins
            )
        else:
            AD = np.fft.fft(
                merge_RD * azimuth_win[:, np.newaxis, np.newaxis],
                axis=0,
                n=self.azimuthBins
            )
        AD = np.fft.fftshift(AD, axes=0)
        AD = np.mean(
            AD[:, :, self.crop_num:-self.crop_num],
            axis=2,
            keepdims=False
        )
        AD = np.concatenate(
            [np.expand_dims(np.real(AD), axis=-1),
             np.expand_dims(np.imag(AD), axis=-1)],
            axis=-1
        )

        return RA, RD, AD

    def saveRadarData(self, save_data, save_path, idxFrame):
        AD_save_path = os.path.join(save_path, "raw_frame_AD")
        RA_save_path = os.path.join(save_path, "raw_frame_RA")
        RD_save_path = os.path.join(save_path, "raw_frame_RD")
        self.check_path([save_path, AD_save_path, RA_save_path, RD_save_path])
        RA, RD, AD = save_data
        RA = RA.astype(np.float32)
        RD = RD.astype(np.float32)
        AD = AD.astype(np.float32)
        np.save(os.path.join(AD_save_path, f"{idxFrame:09d}.npy"), AD)
        np.save(os.path.join(RD_save_path, f"{idxFrame:09d}.npy"), RD)
        for RA_idx in range(RA.shape[0]):
            if RA_idx + 1 in self.chirp_list:
                RA_item = RA[RA_idx, ...]
                np.save(
                    os.path.join(RA_save_path, f"{idxFrame:03d}_{RA_idx + 1:09d}.npy"),
                    RA_item
                )

    # === 新增：单帧处理函数，供多线程调用 ===
    def _process_single_frame(self, idxFrame, frameAzimuth, save_dir):
        outputAzimuth = self.generateHeatmap(frameAzimuth)
        self.saveRadarData(outputAzimuth, save_dir, idxFrame)

    # === 多线程版本 ===
    def processRadarDataAzimuth(self, num_workers=None):
        if num_workers is None:
            num_workers = os.cpu_count() or 4

        for idxName in tqdm(range(len(self.radarDataFileNameGroup))):
            # Azimuth
            if 'Azimuth' in self.create_type:
                adcDataAzimuths = self.getAdcDataFromDCA1000(
                    self.radarDataFileNameGroup[idxName][0]
                )
                adcDataAzimuths = adcDataAzimuths.reshape(
                    self.numRX, -1, self.numTX * self.idxProcChirp, self.numADCSamples
                )
                adcDataAzimuths = np.split(
                    adcDataAzimuths, adcDataAzimuths.shape[1], axis=1
                )

                save_dir_az = os.path.join(self.saveDirNameGroup[idxName], 'azimuth')
                self.check_path(save_dir_az)

                # 多线程处理每一帧
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for idxFrame in range(len(adcDataAzimuths)):
                        frameAzimuth = adcDataAzimuths[idxFrame].squeeze()
                        futures.append(
                            executor.submit(
                                self._process_single_frame,
                                idxFrame,
                                frameAzimuth,
                                save_dir_az
                            )
                        )

                    # 用 tqdm 包一下 futures 完成情况
                    for _ in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Seq {idxName + 1} Azimuth"
                    ):
                        pass

            # Elevation（如果需要，可以同样并行）
            if 'Elevation' in self.create_type:
                adcDataElevation = self.getAdcDataFromDCA1000(
                    self.radarDataFileNameGroup[idxName][1]
                )
                adcDataElevation = adcDataElevation.reshape(
                    self.numRX, -1, self.numTX * self.idxProcChirp, self.numADCSamples
                )
                adcDataElevation = np.split(
                    adcDataElevation, adcDataElevation.shape[1], axis=1
                )

                save_dir_el = os.path.join(self.saveDirNameGroup[idxName], 'elevation')
                self.check_path(save_dir_el)

                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    for idxFrame in range(len(adcDataElevation)):
                        frameEl = adcDataElevation[idxFrame].squeeze()
                        futures.append(
                            executor.submit(
                                self._process_single_frame,
                                idxFrame,
                                frameEl,
                                save_dir_el
                            )
                        )

                    for _ in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Seq {idxName + 1} Elevation"
                    ):
                        pass


if __name__ == '__main__':
    use_filters = [11]
    for use_filter in use_filters:
        radarObject = RadarObject()
        radarObject.processRadarDataAzimuth(num_workers=8)