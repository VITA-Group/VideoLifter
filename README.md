<h2 align="center"> <a href="https://arxiv.org/abs/2501.01949">VideoLifter: Lifting Videos to 3D with Fast Hierarchical Stereo Alignment </a>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2501.01949-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.01949) 
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://videolifter.github.io/) 
</h5>

<div align="center">
This repository is the official implementation of VideoLifter, an efficient SfM-free framework for lifting videos into 3D using hierarchical stereo alignment.
</div>
<br>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Data Preparation](#Data Preparation)
  - [Usage](#usage)
- [Citation](#citation)


## Get Started

### Installation
1. Clone InstantSplat and download pre-trained model.
```bash
git clone --recursive https://github.com/VITA-Group/VideoLifter.git
cd VideoLifter
mkdir -p submodules/mast3r/checkpoints/
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P submodules/mast3r/checkpoints/
```

2. Create the environment.
```bash
conda create -n videolifter python=3.10 cmake=3.14.0 -y
conda activate videolifter
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization-confidence
```

3. Optional but highly suggested, compile the cuda kernels for RoPE (as in CroCo v2).
```bash
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
```

### Data preparation
DATAROOT is `./data` by default. Please first make data folder by `mkdir data`.
1. Tanks and Temples

    Download the data preprocessed by [Nope-NeRF](https://github.com/ActiveVisionLab/nope-nerf/?tab=readme-ov-file#Data) as below, and the data is saved into the `./data/Tanks` folder.
    ```bash
    wget https://www.robots.ox.ac.uk/~wenjing/Tanks.zip
    ```

2. CO3D

    We follow [CF-3DGS](https://github.com/NVlabs/CF-3DGS.git) and select the same 10 scenes from CO3D dataset. Download our preprocessed [data](https://drive.google.com/file/d/1Wxo5ukiObHgrvElo25J0zDoNF6mnUWML/view?usp=sharing), and put it saved into the `./data/co3d` folder.

### Usage
For training and evalution on Tanks and Temples, command:
```bash
  bash scripts/train_tt.sh
```
For CO3D, command:
```bash
  bash scripts/train_co3d.sh
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [DUSt3R](https://github.com/naver/dust3r)
- [MASt3R](https://github.com/naver/mast3r)

## Citation
If you find our work useful in your research, please consider giving a star :star: and citing the following paper :pencil:.

```bibTeX
@misc{cong2025videolifter,
      title={VideoLifter: Lifting Videos to 3D with Fast Hierarchical Stereo Alignment}, 
      author={Wenyan Cong and Kevin Wang and Jiahui Lei and Colton Stearns and Yuanhao Cai and Dilin Wang and Rakesh Ranjan and Matt Feiszli and Leonidas Guibas and Zhangyang Wang and Weiyao Wang and Zhiwen Fan},
      year={2025},
      eprint={2501.01949},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.01949}, 
}