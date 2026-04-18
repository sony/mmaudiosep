# MMAudioSep: Taming Video-to-Audio Generative Model Towards Video/Text-Queried Sound Separation

## MMAudioSep - Official PyTorch Implementation

> **MMAudioSep: Taming Video-to-Audio Generative Model Towards Video/Text-Queried Sound Separation**<br>
> [Akira Takahashi](https://scholar.google.co.jp/citations?user=oKUpOaQAAAAJ), [Shusuke Takahashi](https://scholar.google.co.jp/citations?user=_mhxayYAAAAJ), [Yuki Mitsufuji](https://www.yukimitsufuji.com/)<br>
> Sony Group Corporation<br>
> In ICASSP 2026. <br>
> [arXiv](https://arxiv.org/abs/2510.09065)<br>

## Installation

This software has only been tested on Ubuntu.

### Prerequisites

We recommend using a [miniforge](https://github.com/conda-forge/miniforge) environment.

- Python 3.9+
- PyTorch **2.5.1+** and corresponding torchvision/torchaudio (pick your CUDA version https://pytorch.org/, pip install recommended)
<!-- - ffmpeg<7 ([this is required by torchaudio](https://pytorch.org/audio/master/installation.html#optional-dependencies), you can install it in a miniforge environment with `conda install -c conda-forge 'ffmpeg<7'`) -->

**1. Install prerequisite if not yet met:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade
```

(Or any other CUDA versions that your GPUs/driver support)

<!-- ```
conda install -c conda-forge 'ffmpeg<7
```
(Optional, if you use miniforge and don't already have the appropriate ffmpeg) -->

**2. Clone our repository:**

```bash
git clone https://github.com/sony/mmaudiosep.git MMAudioSep
```

**3. Install with pip (install pytorch first before attempting this!):**

```bash
cd MMAudioSep
pip install -e .
```

(If you encounter the File "setup.py" not found error, upgrade your pip with pip install --upgrade pip)


**Pretrained models:**

**Note:** Pretrained models are not yet available, but will be released in an upcoming update.
See [MODELS.md](docs/MODELS.md) for more details.

## Demo

By default, these scripts use the `large_44k` model. 
In our experiments, inference requires approximately 10GB of GPU memory (in 16-bit mode), which should be compatible with most modern GPUs.

### Command-line interface

With `demo.py`

```bash
python demo.py --duration=8 --video=<path to video> --audio=<path to mixture audio> --prompt "your prompt" 
```

The output (audio in `.flac` format, and video in `.mp4` format) will be saved in `./output`.
See the file for more options.
Simply omit the `--video` option for text-query separation.
The default output (and training) duration is 8 seconds. Longer/shorter durations could also work, but a large deviation from the training duration may result in a lower quality.

### Gradio interface

Supports video-query and text-query sound separation.
Use [port forwarding](https://unix.stackexchange.com/questions/115897/whats-ssh-port-forwarding-and-whats-the-difference-between-ssh-local-and-remot) (e.g., `ssh -L 7860:localhost:7860 server`) if necessary. The default port is `7860` which you can specify with `--port`.

```bash
python gradio_demo.py
```

## Training

See [TRAINING.md](docs/TRAINING.md).

## Evaluation

**Note:** Our evaluation code is based on av-benchmark (https://github.com/hkchengrex/av-benchmark).  
It is already usable with the current setup, and we plan to release some modifications soon to better accommodate our specific use case.

See [EVAL.md](docs/EVAL.md).

## Training Datasets

MMAudioSep was trained on several datasets, including [AudioSet](https://research.google.com/audioset/), [Freesound](https://github.com/LAION-AI/audio-dataset/blob/main/laion-audio-630k/README.md), [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/), [AudioCaps](https://audiocaps.github.io/), and [WavCaps](https://github.com/XinhaoMei/WavCaps). These datasets are subject to specific licenses, which can be accessed on their respective websites. We do not guarantee that the pre-trained models are suitable for commercial use. Please use them at your own risk.

## Citation
If you find this work useful for your research, please cite our paper as follows:

```bibtex
@inproceedings{takahashi2025mmaudiosep,
  title={{MMAudioSep}: Taming Video-to-Audio Generative Model Towards Video/Text-Queried Sound Separation},
  author={Takahashi, Akira and Takahashi, Shusuke  and Mitsufuji, Yuki},
  booktitle={ICASSP},
  year={2026}
}
```

## Relevant Repositories

- [MMAudio](https://github.com/hkchengrex/mmaudio) — This repository serves as the foundation for the main codebase. Our implementation is based on MMAudio, with additional modifications and extensions tailored to our use case.
- [av-benchmark](https://github.com/hkchengrex/av-benchmark) — for benchmarking results.

## Acknowledgement

We would like to express our gratitude to:
- [Make-An-Audio 2](https://github.com/bytedance/Make-An-Audio-2) for the 16kHz BigVGAN pretrained model and the VAE architecture
- [BigVGAN](https://github.com/NVIDIA/BigVGAN)
- [Synchformer](https://github.com/v-iashin/Synchformer) 
- [EDM2](https://github.com/NVlabs/edm2) for the magnitude-preserving VAE network architecture
