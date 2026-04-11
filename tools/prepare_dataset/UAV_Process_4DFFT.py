import os
import numpy as np
from tqdm import tqdm
from scipy.signal.windows import taylor



class RadarObject():
    def __init__(self):
        # 路径配置
        self.root = "/home/jackychou/Zhou/Datasets/test_UAV_dataset"
        self.start_single_id = 1
        self.end_single_id = 12

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
        self.elevationBins = 8
        self.crop_num = 12
        self.rangeBins = self.rangeBins + 2 * self.crop_num
        self.use_filter = True

        # 定义文件保存列表 __ 不要动
        self.use_win = True
        self.radarDataFileNameGroup = []
        self.saveDirNameGroup = []
        self.rgbFileNameGroup = []
        self.jointsFileNameGroup = []

        self.initialize(self.start_single_id, self.end_single_id)

    def initialize(self, start_single_id, end_single_id):
        for i in range(start_single_id, end_single_id + 1):
            radarDataFileName = [os.path.join(self.root, f"uav_seqs_{i}", "raw_radar", "azimuth"),
                                 os.path.join(self.root, f"uav_seqs_{i}", "raw_radar", "elevation")]
            if self.use_filter:
                saveDirName = os.path.join(self.root, f"uav_seqs_{i}", "4dfft_slice_frame_1")
            else:
                saveDirName = os.path.join(self.root, f"uav_seqs_{i}", "4dfft_slice_frame_0")
            self.check_path(saveDirName)
            self.radarDataFileNameGroup.append(radarDataFileName)
            self.saveDirNameGroup.append(saveDirName)


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


    def generateHeatmap(self, frame):
        data_first = frame[:, 0::3, :]
        data_second = frame[:, 1::3, :]
        data_third = frame[:, 2::3, :]
        # Azimuth-Doppler-Range
        data_merge = np.concatenate([data_first, data_second, data_third], axis=0)

        # 杂波滤除 -- 针对doppler维度 wait for write
        if self.use_filter:
            data_merge = data_merge - np.mean(data_merge, axis=1, keepdims=True)

        # Range-FFT
        if self.use_win:
            range_win = np.hamming(self.numADCSamples)
            merge_range = np.fft.fft(data_merge * range_win[np.newaxis, np.newaxis, :], axis=2, n=self.rangeBins)
        else:
            merge_range = np.fft.fft(data_merge, axis=2, n=self.rangeBins)

        # create RD
        merge_RD = np.fft.fft(merge_range, axis=1, n=self.dopplerBins)
        merge_RD = np.fft.fftshift(merge_RD, axes=1)
        merge_RD_azimuth = np.expand_dims(np.concatenate([merge_RD[0::3, ...], merge_RD[2::3, ...]], axis=0), axis=0)
        merge_RD_elevation = np.expand_dims(merge_RD[1::3, ...], axis=0)

        if self.use_win:
            # jianhong
            azimuth_win_8 = taylor(8)
            azimuth_win_4 = taylor(4)
            elevation_win = taylor(2)
            merge_RDA_azimuth = np.fft.fft(merge_RD_azimuth * azimuth_win_8[np.newaxis, :, np.newaxis, np.newaxis], axis=1, n=self.azimuthBins)
            merge_RDA_azimuth = np.fft.fftshift(merge_RDA_azimuth, axes=1)
            merge_RDA_elevation = np.fft.fft(merge_RD_elevation * azimuth_win_4[np.newaxis, :, np.newaxis, np.newaxis], axis=1, n=self.azimuthBins)
            merge_RDA_elevation = np.fft.fftshift(merge_RDA_elevation, axes=1)
            merge_RDA = np.concatenate([merge_RDA_azimuth, merge_RDA_elevation], axis=0)
            merge_RDEA = np.fft.fft(merge_RDA * elevation_win[:, np.newaxis, np.newaxis, np.newaxis], axis=0, n=self.elevationBins)
            merge_RDEA = np.fft.fftshift(merge_RDEA, axes=0)
        else:
            # ziyi
            merge_RD_elevation = np.pad(merge_RD_elevation, ((0, 0), (2, 2), (0, 0), (0, 0)), mode='constant')
            merge_RDEA = np.concatenate([merge_RD_azimuth, merge_RD_elevation], axis=0)
            merge_RDEA = np.pad(merge_RDEA, ((0, self.elevationBins - 2), (0, self.azimuthBins - self.numTX * self.numRX + self.numRX), (0, 0), (0, 0)), mode='constant')
            for idxChirp in range(self.idxProcChirp):
                for idxADC in range(self.numADCSamples):
                    merge_RDEA[:, 2, idxChirp, idxADC] = np.fft.fft(merge_RDEA[:, 2, idxChirp, idxADC], n=self.elevationBins)
                    merge_RDEA[:, 3, idxChirp, idxADC] = np.fft.fft(merge_RDEA[:, 3, idxChirp, idxADC], n=self.elevationBins)
                    merge_RDEA[:, 4, idxChirp, idxADC] = np.fft.fft(merge_RDEA[:, 4, idxChirp, idxADC], n=self.elevationBins)
                    merge_RDEA[:, 5, idxChirp, idxADC] = np.fft.fft(merge_RDEA[:, 5, idxChirp, idxADC], n=self.elevationBins)
                    for idxEle in range(self.elevationBins):
                        merge_RDEA[idxEle, :, idxChirp, idxADC] = np.fft.fft(merge_RDEA[idxEle, :, idxChirp, idxADC], n=self.azimuthBins)

        # Elevation-Azimuth-Doppler-Range

        merge_RDEA = np.mean(merge_RDEA[..., self.crop_num:-self.crop_num], axis=0, keepdims=False)
        # Azimuth-Doppler-Range
        RD = np.mean(merge_RDEA, axis=0, keepdims=False)
        RD = np.transpose(RD, axes=[1, 0])
        RD = np.concatenate([np.expand_dims(np.real(RD), axis=-1), np.expand_dims(np.imag(RD), axis=-1)], axis=-1)

        RA = np.transpose(merge_RDEA, axes=[1, 2, 0])
        RA = np.concatenate([np.expand_dims(np.real(RA), axis=-1), np.expand_dims(np.imag(RA), axis=-1)], axis=-1)

        AD = np.mean(merge_RDEA, axis=2, keepdims=False)
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
            RA_item = RA[RA_idx, ...]
            np.save(os.path.join(RA_save_path, f"{idxFrame:03d}_{RA_idx+1:09d}.npy"), RA_item)


    def processRadarDataAzimuth(self):
        for idxName in tqdm(range(len(self.radarDataFileNameGroup))):
            # TX all Range
            adcDataAzimuths = self.getAdcDataFromDCA1000(self.radarDataFileNameGroup[idxName][0])
            adcDataAzimuths = adcDataAzimuths.reshape(self.numRX, -1, self.numTX * self.idxProcChirp, self.numADCSamples)
            adcDataAzimuths = np.split(adcDataAzimuths, adcDataAzimuths.shape[1], axis=1)

            for idxFrame in tqdm(range(len(adcDataAzimuths))):
                frameAzimuth = adcDataAzimuths[idxFrame].squeeze()
                outputAzimuth = self.generateHeatmap(frameAzimuth)
                self.saveRadarData(outputAzimuth, os.path.join(self.saveDirNameGroup[idxName], 'azimuth'), idxFrame)



if __name__ == '__main__':
    radarObject = RadarObject()
    radarObject.processRadarDataAzimuth()