## A corrected implementation of GMAN with PyTorch

GMAN model has become a very frequently used model for traffic predictions. Based on a previous release of GMAN implementation with PyTorch ("https://github.com/VincLee8188/GMAN-PyTorch/tree/master"), I spent a week to go through the code. While the code looks good overall, there are a number of issues in the code. So, this repository is a corrected and cleaned version of the implementaion. It seems like the original repository holders are not reviewing the code anymore.

## Improvements
- More comprehensive comments
- Add type annotations
- Support multiple torch devices ('cpu', 'cuda', 'mps')
- Refactored/Improved pipeline (e.g., better data loader)

## Basic Requirements
* Python (>=3.11 recommended)
* PyTorch
* Pandas
* matplotlib
* numpy
* pytest

### Usage
Run the whole training process
```
python main.py
```

Run basic tests
```
python model_test.py
```

## Dataset
The datasets could be  unzipped and load from the data directory in this repository.

## Citation
This version of implementation is only for learning purpose. For research, please refer to and cite from the work original authors:
```
@inproceedings{zheng2020gman,
  title={Gman: A graph multi-attention network for traffic prediction},
  author={Zheng, Chuanpan and Fan, Xiaoliang and Wang, Cheng and Qi, Jianzhong},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={34},
  number={01},
  pages={1234--1241},
  year={2020}
}
```
