<div align="center">

<p align="center">
  <img src="assets/logo.png" width="20%"/>
</p>

# NaVILA: Legged Robot Vision-Language-Action Model for Navigation (RSS'25)

[![website](https://img.shields.io/badge/website-6DE1D2?style=for-the-badge&logo=safari&labelColor=555555)](https://navila-bot.github.io/)
[![Arxiv](https://img.shields.io/badge/Arxiv-F75A5A?style=for-the-badge&logo=arxiv&labelColor=555555)](https://arxiv.org/abs/2412.04453)
[![Huggingface](https://img.shields.io/badge/Huggingface-FFD63A?style=for-the-badge&logo=huggingface&labelColor=555555)](https://huggingface.co/collections/a8cheng/navila-legged-robot-vision-language-action-model-for-naviga-67cfc82b83017babdcefd4ad)
[![Locomotion Code](https://img.shields.io/badge/Locomotion%20Code%20-FFA955?style=for-the-badge&logo=github&labelColor=555555)](https://github.com/yang-zj1026/legged-loco)

 
<p align="center">
  <img src="assets/teaser.gif" width="600">
</p>

</div>

> [!IMPORTANT]
> This repository is forked from [NaVILA](https://github.com/AnjieCheng/NaVILA).
> See the original repository !! \
> This repository contains the pyproject.toml and .gitignore files, which are made to **support package management with uv**.

## 💡 Introduction

NaVILA is a two-level framework that combines VLAs with locomotion skills for navigation. It generates high-level language-based commands, while a real-time locomotion policy ensures obstacle avoidance.

<p align="center">
  <img src="assets/method.png" width="600">
</p>


## TODO
- [x] Release mode/weight/evaluation.
- [x] Release training code. (around June 30th)
- [x] Release YouTube Human Touring dataset. (around June 30th)
- [x] Release Isaac Sim evaluation, please see [here](https://github.com/yang-zj1026/NaVILA-Bench).

## 🔧 Jetson AGX Orin (JetPack 7.2) — uv sync setup

This fork supports a **`uv sync`-only inference environment** on **Jetson AGX Orin / JetPack 7.2 (L4T r39.2) / Ubuntu 24.04 / aarch64 / sm_87**. No conda/pip and no manual `transformers_replace` file-copy patching are required for inference.

### Prerequisites
- Flash the device with the **NVIDIA SDK Manager** so the standard JetPack 7.2 BSP/driver stack is installed. (A plain Ubuntu install via the Canonical guide leaves the GPU driver/library stack inconsistent and CUDA fails with `cuInit → 801, operation not supported`.) `cat /etc/nv_tegra_release` should show `R39 ... REVISION: 2.0`.
- Add your user to the `video` and `render` groups for GPU device access, then **log out/in (or reboot)**:
  ```bash
  sudo usermod -aG video,render $USER
  ```

### Install
```bash
# 1. uv (it manages its own Python 3.12 via python-build-standalone)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2. sync — on aarch64 torch/torchvision resolve from the SBSA cu132 index
uv sync
```

### Verify
```bash
# GPU is recognized on sm_87
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_capability())"
# True (8, 7)

# End-to-end inference — loads a8cheng/navila-llama3-8b-8f (4-bit, fp16, CUDA)
uv run python examples/model_test.py
# -> "The next action is move forward 25 cm."
```

### Notes / known caveats
- **torch**: pulled from the SBSA `cu132` index (`download.pytorch.org/whl/cu132`) gated on `platform_machine=='aarch64'`. The `Found GPU0 Orin ... CC 8.7` build-exclusion warning is expected — inference runs via PTX fallback (NVIDIA forum 372773, POST 3). x86_64 dev machines keep resolving torch from PyPI.
- **bitsandbytes** ⚠️: 0.49.x ships **no CUDA 13.2 binary** (only up to `cuda130`), so 4-bit (NF4) quantization uses the minor-compatible CUDA 13.0 binary via `BNB_CUDA_VERSION=130`. This is applied **automatically** by `llava/__init__.py` whenever torch reports CUDA 13.2 — no manual `export` needed. Drop the workaround once bitsandbytes ships a `cuda132` wheel ([bitsandbytes #1937](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1937)).
- **decord**: dropped from base deps (no aarch64 wheel) and lazily imported — only needed for video decoding.
- **flash-attn / DeepSpeed**: not required for SigLIP/Llama-3 inference. `deepspeed.comm` falls back to `torch.distributed`, and the `flash_attn` import is guarded, so `import llava` works without them.
- **Training / VLN-CE evaluation** still follow the conda flow below (out of scope for `uv sync`).

## 🚀 Training
### Installation
To build environment for training NaVILA, please run the following:
```bash
./environment_setup.sh navila
conda activate navila
```
Optional: If you plan to use TensorBoard for logging, install `tensorboardX` via pip.

### Dataset
For general VQA datasets like `video_chatgpt`, `sharegpt_video`, `sharegpt4v_sft`, please follow the data preparation instructions in [NVILA](https://github.com/NVlabs/VILA).
We provide annotations for `envdrop`, `scanqa`, `r2r`, `rxr`, and `human` on [Hugging Face](https://huggingface.co/datasets/a8cheng/NaVILA-Dataset).
Please download the repo and extract the `tar.gz` files in their respective subfolders. 
<p align="center">
<img src="assets/human_touring.gif" width="600">
</p>

* **YouTube Human Touring:**  
Due to copyright restrictions, raw videos/images are not released. We provide **[video IDs](https://huggingface.co/datasets/a8cheng/NaVILA-Dataset/blob/main/Human/video_ids.txt)** and **annotations**. You can download the videos using `yt-dlp` and extract frames using: `scripts/extract_rawframes.py`

* **EnvDrop:**  
Due to the large number of videos, we provide **annotations only**. Please download the **R2R augmented split** from [R2R_VLNCE_v1-3_preprocessed.zip](https://drive.google.com/file/d/1fo8F4NKgZDH-bPSdVU3cONAkt5EW-tyr/view?usp=sharing) and render corresponding videos using [VLN-CE](https://github.com/jacobkrantz/VLN-CE).

The data should have structure like:
```graphql
NaVILA-Dataset
├─ EnvDrop
|   ├─ videos
|   |    ├─ 1.mp4
|   |    ├─ ...
|   ├─ annotations.json
├─ Human
|   ├─ raw_frames
|   |    ├─ Aei0GpsWNys
|   |    |    ├─ 0001.jpg
|   |    |    ├─ ...
|   |    ├─ ...
|   ├─ videos
|   |    ├─ Aei0GpsWNys.mp4
|   |    ├─ ...
|   ├─ annotations.json
|   ├─ video_ids.txt
├─ R2R
|   ├─ train
|   |    ├─ 1
|   |    |    ├─ frame_0.jpg 
|   |    |    ├─ ...
|   |    ├─ ...
|   ├─ annotations.json
├─ RxR
|   ├─ train
|   |    ├─ 1
|   |    |    ├─ frame_0.jpg 
|   |    |    ├─ ...
|   |    ├─ ...
|   ├─ annotations.json
├─ ScanQA
|   ├─ videos
|   |    ├─ scene0760_00.mp4
|   |    ├─ ...
|   ├─ annotations
|   |    ├─ ScanQA_v1.0_train_reformat.json
|   |    ├─ ...
```


### Training
The pretrain model to start from is provided in [a8cheng/navila-siglip-llama3-8b-v1.5-pretrain](https://huggingface.co/a8cheng/navila-siglip-llama3-8b-v1.5-pretrain). Please modify the data paths in `llava/data/datasets_mixture.py` and use the script in `scripts/train/sft_8frames.sh` to lanuch the training. 


## 📊 Evaluation

### Installation

This repository builds on [VLN-CE](https://github.com/jacobkrantz/VLN-CE), which relies on older versions of [Habitat-Lab](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7) and [Habitat-Sim](https://github.com/facebookresearch/habitat-lab/tree/v0.1.7). The installation process requires several modifications and can be complex.

1. Create a Conda Environment with Python 3.10
```bash
conda create -n navila-eval python=3.10
conda activate navila-eval
```

2. Build Habitat-Sim & Lab (v0.1.7) from **Source**

Follow the [VLN-CE setup guide](https://github.com/jacobkrantz/VLN-CE?tab=readme-ov-file#setup).
To resolve NumPy compatibility issues, apply the following hotfix:
```bash
python evaluation/scripts/habitat_sim_autofix.py # replace habitat_sim/utils/common.py
```

3. Install VLN-CE Dependencies
```bash
pip install -r evaluation/requirements.txt
```

4. Install VILA Dependencies
```bash
# Install FlashAttention2
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install VILA (assum in root dir)
pip install -e .
pip install -e ".[train]"
pip install -e ".[eval]"

# Install HF's Transformers
pip install git+https://github.com/huggingface/transformers@v4.37.2
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/transformers_replace/* $site_pkg_path/transformers/
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/
```

5. Fix WebDataset Version for VLN-CE Compatibility
```bash
pip install webdataset==0.1.103
```

### Data
Please follow [VLN-CE](https://github.com/jacobkrantz/VLN-CE) and download R2R and RxR annotations, and scene data inside the `evaluation/data` folder. The data should have structure like:
```graphql
data/datasets
├─ RxR_VLNCE_v0
|   ├─ train
|   |    ├─ train_guide.json.gz
|   |    ├─ ...
|   ├─ val_unseen
|   |    ├─ val_unseen_guide.json.gz
|   |    ├─ ...
|   ├─ ...
├─ R2R_VLNCE_v1-3_preprocessed
|   ├─ train
|   |    ├─ train.json.gz
|   |    ├─ ...
|   ├─ val_unseen
|   |    ├─ val_unseen.json.gz
|   |    ├─ ...
data/scene_datasets
├─ mp3d
|   ├─ 17DRP5sb8fy
|   |    ├─ 17DRP5sb8fy.glb
|   |    ├─ ...
|   ├─ ...
```
### Running Evaluation
1. Download the checkpoint from [a8cheng/navila-llama3-8b-8f](https://huggingface.co/a8cheng/navila-llama3-8b-8f).
2. Run evaluation on R2R using:
```bash
cd evaluation
bash scripts/eval/r2r.sh CKPT_PATH NUM_CHUNKS CHUNK_START_IDX "GPU_IDS"
```
Examples:
* Single GPU:
    ```bash
    bash scripts/eval/r2r.sh CKPT_PATH 1 0 "0"
    ```
* Multiple GPUs (e.g., 8 GPUs):
    ```bash
    bash scripts/eval/r2r.sh CKPT_PATH 8 0 "0,1,2,3,4,5,6,7"
    ```
3. Visualized videos are saved in 
```bash
./eval_out/CKPT_NAME/VLN-CE-v1/val_unseen/videos
```
<p align="center">
  <img src="assets/sample.gif" width="600">
</p>
4. Aggregate results and view the scores

```bash
python scripts/eval_jsons.py ./eval_out/CKPT_NAME/VLN-CE-v1/val_unseen NUM_CHUNKS
```

_______________________________________________________________

## 📜 Citation

```bibtex
@inproceedings{cheng2025navila,
        title={Navila: Legged robot vision-language-action model for navigation},
        author={Cheng, An-Chieh and Ji, Yandong and Yang, Zhaojing and Gongye, Zaitian and Zou, Xueyan and Kautz, Jan and B{\i}y{\i}k, Erdem and Yin, Hongxu and Liu, Sifei and Wang, Xiaolong},
        booktitle={RSS},
        year={2025}
}
```
