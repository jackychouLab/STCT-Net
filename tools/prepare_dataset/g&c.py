from scipy.signal.windows import taylor
from scipy.signal import butter, lfilter
import os
import numpy as np
from tools.instruments.instruments import read_npy, read_mat
from tqdm import tqdm
import subprocess


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
        self.root = "/home/jackychou/Zhou/Datasets/UAV1.0"
        self.save_root = "/home/jackychou/Datasets/UAV1.0"
        self.start_single_id = -1
        self.end_single_id = -1
        self.idx_list = trainIdx
        self.create_type = ['Azimuth'] # 'Azimuth', 'Elevation'

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
        self.dopplerBins = 256
        self.azimuthBins = 128
        self.crop_num = 12
        self.rangeBins = self.rangeBins + 2 * self.crop_num
        self.use_filter = use_filter

        # 定义文件保存列表
        self.radarDataFileNameGroup = []
        self.saveDirNameGroup = []
        self.rgbFileNameGroup = []
        self.jointsFileNameGroup = []
        self.initialize()

        chirp_list = [
            # [1, 85, 170, 255],
            # [1, 37, 73, 109, 46, 182, 218, 255],
            # [1, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255],
            # [1, 9, 17, 25, 33, 41, 50, 58, 66, 74, 82, 91, 99, 107, 115, 123, 132, 140, 148, 156, 164, 173, 181, 189,
            #  197, 205, 214, 222, 230, 238, 246, 255],
            [i for i in range(1, 256)]
        ]

        self.chirp_list = merge_creat_new(chirp_list)

        assert self.use_filter in [0, 1, 2, 3, 4, 10, 11, 22, 33, 44]


    def initialize(self):
        for i in self.idx_list:
            radarDataFileName = [os.path.join(self.root, f"uav_seqs_{i}", "raw_radar", "azimuth"), os.path.join(self.root, f"uav_seqs_{i}", "raw_radar", "elevation")]
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
        if type(x) is str:
            x = [x]
        for x_i in x:
            if os.path.exists(x_i) is False:
                os.mkdir(x_i)


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
        if self.use_filter == 2 or self.use_filter == 4 or self.use_filter == 22 or self.use_filter == 44:
            for temp_idx in range(data_merge.shape[0]):
                for temp_jdx in range(data_merge.shape[1]):
                    data_merge[temp_idx, temp_jdx, :] = butter_lowpass_filter(data_merge[temp_idx, temp_jdx, :], 6)

        if self.use_filter == 1 or self.use_filter == 4 or self.use_filter == 11 or self.use_filter == 44:
            data_merge = data_merge - np.mean(data_merge, axis=1, keepdims=True)

        # Range-FFT
        range_win = np.hamming(self.numADCSamples)
        merge_range = np.fft.fft(data_merge * range_win[np.newaxis, np.newaxis, :], axis=2, n=self.rangeBins)

        if self.use_filter == 3 or self.use_filter == 33:
            for temp_idx in range(data_merge.shape[0]):
                for temp_jdx in range(data_merge.shape[1]):
                    data_merge[temp_idx, temp_jdx, :] = butter_lowpass_filter(data_merge[temp_idx, temp_jdx, :], 6)

        # create RD
        merge_RD = np.fft.fft(merge_range, axis=1, n=self.dopplerBins)
        merge_RD = np.fft.fftshift(merge_RD, axes=1)
        RD = np.mean(merge_RD[:, :, self.crop_num:-self.crop_num], axis=0, keepdims=False)
        RD = np.transpose(RD, axes=[1, 0])
        RD = np.concatenate([np.expand_dims(np.real(RD), axis=-1), np.expand_dims(np.imag(RD), axis=-1)], axis=-1)

        # create RA
        azimuth_win = taylor(self.numRX * self.numTX)
        if self.use_filter == 11 or self.use_filter == 22 or self.use_filter == 33 or self.use_filter == 44 or self.use_filter == 10:
            azimuth_win = taylor(self.numRX * 2)
            RA = np.fft.fft(np.concatenate([merge_range[0:4, ...], merge_range[-4:, ...]], axis=0) * azimuth_win[:, np.newaxis, np.newaxis], axis=0, n=self.azimuthBins)
        else:
            RA = np.fft.fft(merge_range * azimuth_win[:, np.newaxis, np.newaxis], axis=0, n=self.azimuthBins)
        RA = np.fft.fftshift(RA, axes=0)
        RA = RA[:, :, self.crop_num:-self.crop_num]
        RA = np.transpose(RA, axes=[1, 2, 0])
        RA = np.concatenate([np.expand_dims(np.real(RA), axis=-1), np.expand_dims(np.imag(RA), axis=-1)], axis=-1)

        # create AD
        if self.use_filter == 11 or self.use_filter == 22 or self.use_filter == 33 or self.use_filter == 44 or self.use_filter == 10:
            azimuth_win = taylor(self.numRX * 2)
            AD = np.fft.fft(np.concatenate([merge_RD[0:4, ...], merge_RD[-4:, ...]], axis=0) * azimuth_win[:, np.newaxis, np.newaxis], axis=0, n=self.azimuthBins)
        else:
            AD = np.fft.fft(merge_RD * azimuth_win[:, np.newaxis, np.newaxis], axis=0, n=self.azimuthBins)
        AD = np.fft.fftshift(AD, axes=0)
        AD = np.mean(AD[:, :, self.crop_num:-self.crop_num], axis=2, keepdims=False)
        AD = np.concatenate([np.expand_dims(np.real(AD), axis=-1), np.expand_dims(np.imag(AD), axis=-1)], axis=-1)

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
                np.save(os.path.join(RA_save_path, f"{idxFrame:03d}_{RA_idx+1:09d}.npy"), RA_item)


    def processRadarDataAzimuth(self):
        for idxName in tqdm(range(len(self.radarDataFileNameGroup))):
            if 'Azimuth' in self.create_type:
                # TX all Range
                adcDataAzimuths = self.getAdcDataFromDCA1000(self.radarDataFileNameGroup[idxName][0])
                adcDataAzimuths = adcDataAzimuths.reshape(self.numRX, -1, self.numTX * self.idxProcChirp, self.numADCSamples)

                adcDataAzimuths = np.split(adcDataAzimuths, adcDataAzimuths.shape[1], axis=1)
                for idxFrame in tqdm(range(len(adcDataAzimuths))):
                    frameAzimuth = adcDataAzimuths[idxFrame].squeeze()
                    outputAzimuth = self.generateHeatmap(frameAzimuth)
                    self.saveRadarData(outputAzimuth, os.path.join(self.saveDirNameGroup[idxName], 'azimuth'), idxFrame)

            if 'Elevation' in self.create_type:
                adcDataAzimuths = self.getAdcDataFromDCA1000(self.radarDataFileNameGroup[idxName][1])
                adcDataAzimuths = adcDataAzimuths.reshape(self.numRX, -1, self.numTX * self.idxProcChirp, self.numADCSamples)

                adcDataAzimuths = np.split(adcDataAzimuths, adcDataAzimuths.shape[1], axis=1)
                for idxFrame in tqdm(range(len(adcDataAzimuths))):
                    frameAzimuth = adcDataAzimuths[idxFrame].squeeze()
                    outputAzimuth = self.generateHeatmap(frameAzimuth)
                    self.saveRadarData(outputAzimuth, os.path.join(self.saveDirNameGroup[idxName], 'elevation'), idxFrame)


if __name__ == '__main__':
    import time
    time.sleep(27000)
    use_filters = [11]
    for use_filter in use_filters:
        root = "/home/jackychou/dataset/UAV-Radar"
        trainIdx = [1, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62,
                    64, 66, 67, 68, 69, 70, 72, 75, 76, 78, 79]
        trainDataType = 'python'  # 'matlab' 'python' '4dfft'
        use_cluster_filter = use_filter
        radarType = 'azimuth'  # 'azmimuth' 'elevation'

        # radarObject = RadarObject()
        # radarObject.processRadarDataAzimuth()

        # real imag num
        raMean = [0., 0., 0., 0., 0.]
        rdMean = [0., 0., 0., 0., 0.]
        adMean = [0., 0., 0., 0., 0.]
        raStd = [0., 0., 0., 0., 0.]
        rdStd = [0., 0., 0., 0., 0.]
        adStd = [0., 0., 0., 0., 0.]

        # 计算均值
        for idx in tqdm(trainIdx):
            seqPath = os.path.join(root, f'uav_seqs_{idx}', f'{trainDataType}_slice_frame_{use_cluster_filter}', radarType)
            raPath = os.path.join(seqPath, 'raw_frame_RA')
            rdPath = os.path.join(seqPath, 'raw_frame_RD')
            adPath = os.path.join(seqPath, 'raw_frame_AD')
            # ra
            packagePath = raPath
            fileNameList = os.listdir(packagePath)
            for FileName in fileNameList:
                filePath = os.path.join(packagePath, FileName)
                if trainDataType == "matlab":
                    data = read_mat(filePath)
                else:
                    data = read_npy(filePath)
                data_real = data[..., 0]
                data_imag = data[..., 1]
                data_magnitude = np.abs(data[..., 0] + data[..., 1] * 1j)
                data_phase = np.arctan2(np.imag(data[..., 1]), np.real(data[..., 0]))
                raMean[0] += np.sum(data_real)
                raMean[1] += np.sum(data_imag)
                raMean[2] += np.sum(data_magnitude)
                raMean[3] += np.sum(data_phase)
                raMean[4] += data_real.reshape(-1).shape[0]

            # rd
            packagePath = rdPath
            fileNameList = os.listdir(packagePath)
            for FileName in fileNameList:
                filePath = os.path.join(packagePath, FileName)
                if trainDataType == "matlab":
                    data = read_mat(filePath)
                else:
                    data = read_npy(filePath)
                data_real = data[..., 0]
                data_imag = data[..., 1]
                data_magnitude = np.abs(data[..., 0] + data[..., 1] * 1j)
                data_phase = np.arctan2(np.imag(data[..., 1]), np.real(data[..., 0]))
                rdMean[0] += np.sum(data_real)
                rdMean[1] += np.sum(data_imag)
                rdMean[2] += np.sum(data_magnitude)
                rdMean[3] += np.sum(data_phase)
                rdMean[4] += data_real.reshape(-1).shape[0]

            # ad
            packagePath = adPath
            fileNameList = os.listdir(packagePath)
            for FileName in fileNameList:
                filePath = os.path.join(packagePath, FileName)
                if trainDataType == "matlab":
                    data = read_mat(filePath)
                else:
                    data = read_npy(filePath)
                data_real = data[..., 0]
                data_imag = data[..., 1]
                data_magnitude = np.abs(data[..., 0] + data[..., 1] * 1j)
                data_phase = np.arctan2(np.imag(data[..., 1]), np.real(data[..., 0]))
                adMean[0] += np.sum(data_real)
                adMean[1] += np.sum(data_imag)
                adMean[2] += np.sum(data_magnitude)
                adMean[3] += np.sum(data_phase)
                adMean[4] += data_real.reshape(-1).shape[0]

        raMean[0] = raMean[0] / raMean[4]
        raMean[1] = raMean[1] / raMean[4]
        raMean[2] = raMean[2] / raMean[4]
        raMean[3] = raMean[3] / raMean[4]
        rdMean[0] = rdMean[0] / rdMean[4]
        rdMean[1] = rdMean[1] / rdMean[4]
        rdMean[2] = rdMean[2] / rdMean[4]
        rdMean[3] = rdMean[3] / rdMean[4]
        adMean[0] = adMean[0] / adMean[4]
        adMean[1] = adMean[1] / adMean[4]
        adMean[2] = adMean[2] / adMean[4]
        adMean[3] = adMean[3] / adMean[4]

        # 计算标准差
        for idx in tqdm(trainIdx):
            seqPath = os.path.join(root, f'uav_seqs_{idx}', f'{trainDataType}_slice_frame_{use_cluster_filter}', radarType)
            raPath = os.path.join(seqPath, 'raw_frame_RA')
            rdPath = os.path.join(seqPath, 'raw_frame_RD')
            adPath = os.path.join(seqPath, 'raw_frame_AD')
            # ra
            packagePath = raPath
            fileNameList = os.listdir(packagePath)
            for FileName in fileNameList:
                filePath = os.path.join(packagePath, FileName)
                if trainDataType == "matlab":
                    data = read_mat(filePath)
                else:
                    data = read_npy(filePath)
                data_real = data[..., 0]
                data_imag = data[..., 1]
                data_magnitude = np.abs(data[..., 0] + data[..., 1] * 1j)
                data_phase = np.arctan2(np.imag(data[..., 1]), np.real(data[..., 0]))
                raStd[0] += np.sum((data_real.reshape(-1) - raMean[0]) ** 2)
                raStd[1] += np.sum((data_imag.reshape(-1) - raMean[1]) ** 2)
                raStd[2] += np.sum((data_magnitude.reshape(-1) - raMean[2]) ** 2)
                raStd[3] += np.sum((data_phase.reshape(-1) - raMean[3]) ** 2)
                raStd[4] += data_real.reshape(-1).shape[0]

            # rd
            packagePath = rdPath
            fileNameList = os.listdir(packagePath)
            for FileName in fileNameList:
                filePath = os.path.join(packagePath, FileName)
                if trainDataType == "matlab":
                    data = read_mat(filePath)
                else:
                    data = read_npy(filePath)
                data_real = data[..., 0]
                data_imag = data[..., 1]
                data_magnitude = np.abs(data[..., 0] + data[..., 1] * 1j)
                data_phase = np.arctan2(np.imag(data[..., 1]), np.real(data[..., 0]))
                rdStd[0] += np.sum((data_real.reshape(-1) - rdMean[0]) ** 2)
                rdStd[1] += np.sum((data_imag.reshape(-1) - rdMean[1]) ** 2)
                rdStd[2] += np.sum((data_magnitude.reshape(-1) - rdMean[2]) ** 2)
                rdStd[3] += np.sum((data_phase.reshape(-1) - rdMean[3]) ** 2)
                rdStd[4] += data_real.reshape(-1).shape[0]

            # ad
            packagePath = adPath
            fileNameList = os.listdir(packagePath)
            for FileName in fileNameList:
                filePath = os.path.join(packagePath, FileName)
                if trainDataType == "matlab":
                    data = read_mat(filePath)
                else:
                    data = read_npy(filePath)
                data_real = data[..., 0]
                data_imag = data[..., 1]
                data_magnitude = np.abs(data[..., 0] + data[..., 1] * 1j)
                data_phase = np.arctan2(np.imag(data[..., 1]), np.real(data[..., 0]))
                adStd[0] += np.sum((data_real.reshape(-1) - adMean[0]) ** 2)
                adStd[1] += np.sum((data_imag.reshape(-1) - adMean[1]) ** 2)
                adStd[2] += np.sum((data_magnitude.reshape(-1) - adMean[2]) ** 2)
                adStd[3] += np.sum((data_phase.reshape(-1) - adMean[3]) ** 2)
                adStd[4] += data_real.reshape(-1).shape[0]

        raStd[0] = np.sqrt(raStd[0] / raStd[4])
        raStd[1] = np.sqrt(raStd[1] / raStd[4])
        raStd[2] = np.sqrt(raStd[2] / raStd[4])
        raStd[3] = np.sqrt(raStd[3] / raStd[4])
        rdStd[0] = np.sqrt(rdStd[0] / rdStd[4])
        rdStd[1] = np.sqrt(rdStd[1] / rdStd[4])
        rdStd[2] = np.sqrt(rdStd[2] / rdStd[4])
        rdStd[3] = np.sqrt(rdStd[3] / rdStd[4])
        adStd[0] = np.sqrt(adStd[0] / adStd[4])
        adStd[1] = np.sqrt(adStd[1] / adStd[4])
        adStd[2] = np.sqrt(adStd[2] / adStd[4])
        adStd[3] = np.sqrt(adStd[3] / adStd[4])

        if os.path.exists("./temp_g_c.txt") is False:
            with open("./temp_g_c.txt", "w") as f:
                pass
        with open("./temp_g_c.txt", "a") as f:
            f.write(f"{use_cluster_filter}\n")
            f.write(f"{trainDataType}_{use_cluster_filter}\n")
            f.write(f"RA:\n实部均值={raMean[0]}\n实部标准差={raStd[0]}\n")
            f.write(f"RA:\n虚部均值={raMean[1]}\n虚部标准差={raStd[1]}\n")
            f.write(f"RA:\n模均值={raMean[2]}\n模标准差={raStd[2]}\n")
            f.write(f"RA:\n相位均值={raMean[3]}\n相位标准差={raStd[3]}\n")
            f.write(f"RD:\n实部均值={rdMean[0]}\n实部标准差={rdStd[0]}\n")
            f.write(f"RD:\n虚部均值={rdMean[1]}\n虚部标准差={rdStd[1]}\n")
            f.write(f"RD:\n模均值={rdMean[2]}\n模标准差={rdStd[2]}")
            f.write(f"RD:\n相位均值={rdMean[3]}\n相位标准差={rdStd[3]}\n")
            f.write(f"AD:\n实部均值={adMean[0]}\n实部标准差={adStd[0]}\n")
            f.write(f"AD:\n虚部均值={adMean[1]}\n虚部标准差={adStd[1]}\n")
            f.write(f"AD:\n模均值={adMean[2]}\n模标准差={adStd[2]}")
            f.write(f"AD:\n相位均值={adMean[3]}\n相位标准差={adStd[3]}\n")

        print(f"{trainDataType}_{use_cluster_filter}")
        print(f"RA:\n实部均值={raMean[0]}\n实部标准差={raStd[0]}")
        print(f"RA:\n虚部均值={raMean[1]}\n虚部标准差={raStd[1]}")
        print(f"RA:\n模均值={raMean[2]}\n模标准差={raStd[2]}")
        print(f"RA:\n相位均值={raMean[3]}\n相位标准差={raStd[3]}")
        print(f"RD:\n实部均值={rdMean[0]}\n实部标准差={rdStd[0]}")
        print(f"RD:\n虚部均值={rdMean[1]}\n虚部标准差={rdStd[1]}")
        print(f"RD:\n模均值={rdMean[2]}\n模标准差={rdStd[2]}")
        print(f"RD:\n相位均值={rdMean[3]}\n相位标准差={rdStd[3]}")
        print(f"AD:\n实部均值={adMean[0]}\n实部标准差={adStd[0]}")
        print(f"AD:\n虚部均值={adMean[1]}\n虚部标准差={adStd[1]}")
        print(f"AD:\n模均值={adMean[2]}\n模标准差={adStd[2]}")
        print(f"AD:\n相位均值={adMean[3]}\n相位标准差={adStd[3]}")

        # root = "/home/jackychou/Datasets/UAV1.0"
        # for idx in tqdm(range(1, 79 + 1)):
        #     FilePath = os.path.join(root, f"uav_seqs_{idx}", f"python_slice_frame_{use_filter}")
        #     param = f'rm -r {FilePath}'
        #     subprocess.run(param, shell=True)







