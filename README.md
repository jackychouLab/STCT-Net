# Towards Low-Altitude UAV Detection in Urban Radar Clutter: a Benchmark and Spatio-Temporal Channel Transfer Network

We are very grateful for the source code provided by [`RODNet`](https://github.com/yizhou-wang/RODNet), which our project extends upon. This is the official implementation of our STCT-Net papers. 

![STCT-Net Overview](./docs/images/1.jpg?raw=true)

Please cite our paper if this repository is helpful for your research:

```
@article{STCT-Net,
  title={Towards Low-Altitude UAV Detection in Urban Radar Clutter: a Benchmark and Spatio-Temporal Channel Transfer Network},
  author={Zhou, Jianhong and Ke, Feng and Zhai, Yikui, and Zheng, XueQiang and Jiang, Ziyi annd Lv, Haolin and Zhang, Xiu Yin},
  journal={Science China-Informtion Sciences},
  volume={-},
  number={-},
  pages={-},
  year={2026},
  publisher={SCIENCE PRESS}
}
```

## Installation

```commandline
cd $STCT-NET_ROOT
git clone https://github.com/jackychouLab/STCT-Net.git
```

Create a conda environment for STCT-Net. Tested under Python 3.10.
```commandline
conda create -n STCT-Net python=3.10 -y
conda activate STCT-Net
```

Note: This work uses CUDA 12.8 and cuDNN 8.9.
```commandline
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Install `cruw-devkit` package. 
```commandline
cd cruw-devkit
pip install -r requirements.txt --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install pynvml fvcore thop timm einops --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.26.4 opencv-python==4.11.0.86 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install . --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
cd ..
```

Setup package.
```commandline
pip install -e . --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation
cd rodnet/ops/tdc_deform_ext
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST="X.X"
pip install -e . --no-build-isolation --no-build-isolation -v
```
**Note:** This work uses an NVIDIA RTX 5000 Ada GPU. Therefore, when setting
```commandline
export TORCH_CUDA_ARCH_LIST="X.X"
 ```
`X.X` should be set to `8.9`. If an NVIDIA RTX 5090 GPU is used instead, `X.X` should be set to `12.0`.

Please adjust this value according to the compute capability of the specific GPU being used.

Download the new CRUW[`Key:gxxg`](https://pan.baidu.com/s/1e8u_0OjR-3g-gToZoPiWiQ) files and use them to replace all files within the compiled CRUW directory.
```commandline
rm -r {Your Environment Path}/lib/python3.10/site-packages/cruw
rm -r {Your Environment Path}/lib/python3.10/site-packages/cruw_devkit-1.1.dist-info
mv $New_cruw {Your Environment Path}/lib/python3.10/site-packages
mv $New_cruw_devkit-1.1.dist-info {Your Environment Path}/lib/python3.10/site-packages
```

## Prepare data for UAVRadar dataset

Download UAVRadar dataset[`Key:6s6v`](https://pan.baidu.com/s/178Fo9nRX2tq0h4-4xu69RA). 
```commandline
cd $UAVRadar_root
cat UAVRadar.tar.gz.part_* | tar -xzvf - -C ./
rm -r UAVRadar.tar.gz.part_*
```

Prepare data and annotations for training.
```commandline
cd $STCT-Net_root/tools/prepare_dataset
```
Use `3_PrepareDataForTrain&Val&Test.py` to prepare. You can generate different data by modifying the `chirp_nuims`, `use_filters` and `sensor_type` parameters.


## Train models

```commandline
python forward_train_UAVRadar.py
```

## Model Weights

The optimal weights of STCT-Net on the UAVRadar dataset are available for download from [`Key:4ux8`](https://pan.baidu.com/s/1_41JZXfCZzIfRiFHd7xUZA).

###### If you encounter any issues with code or data reproduction, please contact me at jackychou_lab@126.com.