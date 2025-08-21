# Robust Radar-Based UAV Detection in Complex Urban Environments via Spatio-temporal Channel Transfer Networks

We are very grateful for the source code provided by [`RODNet`](https://github.com/yizhou-wang/RODNet), which our project extends upon. This is the official implementation of our STCT-Net papers. 

![STCT-Net Overview](./docs/images/1.jpg?raw=true)

Please cite our paper if this repository is helpful for your research:

```
@article{jackychou_stct-net,
  title={Robust Radar-Based UAV Detection in Complex Urban Environments via Spatio-temporal Channel Transfer Networks},
  author={Jianhong Zhou, Feng Ke, Yikui Zhai, Ziyi Jiang, Haolin Lv, Pasquale Coscia, Angelo Genovese, and Xiu Yin Zhang},
  journal={-},
  volume={-},
  number={-},
  pages={-},
  year={-},
  publisher={-}
}
```

## Installation

```commandline
cd $STCT-NET_ROOT
git clone https://github.com/jackychouLab/STCT-Net.git
```

Create a conda environment for STCT-Net. Tested under Python 3.9.
```commandline
conda create -n stctnet python=3.9 -y
conda activate stctnet
```

Install the latest version of  Pytorch for CUDA 12.8.
```commandline
pip3 install torch torchvision
```

Install `cruw-devkit` package. 
Please refer to [`cruw-devit`](https://github.com/yizhou-wang/cruw-devkit) repository for detailed instructions.
```commandline
git clone https://github.com/yizhou-wang/cruw-devkit.git
cd cruw-devkit
pip install .
cd ..
```

Setup package.
```commandline
pip install -e .
```

Download the new CRUW[`Key:qerr`](https://pan.baidu.com/s/1Fu9WDf6TYwRZAkUIGR127w) files and use them to replace all files within the compiled CRUW directory.
```commandline
{Your Environment Path}/lib/python3.9/site-packages/cruw/*
{Your Environment Path}/lib/python3.9/site-packages/cruw_devkit-1.1.dist-info/*
```

## Prepare data for UAV-Radar dataset

Download UAV-Radar dataset[`Key:5tbq`](https://pan.baidu.com/s/1KBas8DEvvH_AukxHmHkfcQ). 

Prepare data and annotations for training.
```commandline
cd STCT-Net/tools/prepare_dataset
```
Use `3_PrepareDataForTrain&Val&Test.py` to prepare. You can generate different data by modifying the `chirp_nuims` and `use_filters` parameters.


## Train models

```commandline
python tools/train.py --config configs/<CONFIG_FILE> \
        --sensor_config cruw-devkit/dataset_configs/<SENSOR_FILE>\
        --data_dir <DATA_FOLDER_NAME> \
        --log_dir logs/<MODEL_NAME>
```

## Inference

```commandline
python tools/test.py --config configs/<CONFIG_FILE> \
        --sensor_config cruw-devkit/dataset_configs/<SENSOR_FILE>\
        --data_dir data/<DATA_FOLDER_NAME> \
        --checkpoint <CHECKPOINT_PATH> \
        --res_dir results/
```
