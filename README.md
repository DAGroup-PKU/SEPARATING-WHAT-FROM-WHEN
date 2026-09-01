<div align="center">

# The Missing Timeline: Temporal Context Routing

### for Script-Aligned Joint Audio-Video Generation

Yichen Liu<sup>1</sup>, Quanwei Zhang<sup>2</sup>, Haozhe Wang<sup>3</sup>, Donghao Zhou<sup>4</sup>, Xiaojie Li, Yang Shi<sup>2</sup>, Jiaming Liu<sup>2</sup>, Ruihua Huang<sup>2</sup>, Yingtian Zou<sup>5</sup>, Daquan Zhou<sup>1</sup>

> <sup>1</sup>Peking University&nbsp;&nbsp;<sup>2</sup>Qwen Business Unit of Alibaba&nbsp;&nbsp;<sup>3</sup>HKUST&nbsp;&nbsp;<sup>4</sup>CUHK&nbsp;&nbsp;<sup>5</sup>Shanghai Jiao Tong University

<p align="center">
  <a href="https://dagroup-pku.github.io/Temporal-Context-Routing.github.io/"><img src="https://img.shields.io/badge/Project-Page-1f8acb"></a>
  <a href="https://arxiv.org"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b"></a>
  <a href="https://huggingface.co/papers"><img src="https://img.shields.io/badge/Hugging%20Face-Daily%20Paper-ffcc4d"></a>
  <a href="https://huggingface.co/starry0929/Separating-What-From-When"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-ffcc4d"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-LTX--2-3fa34d"></a>
</p>

</div>

**Temporal Context Routing (TCR)** is a **training-time, plug-in** pathway that separates *what* a shot or a line describes from *when* it should influence generation. Intervals bypass the text encoder and arrive as a timing term on the text-attention logits, so the video stream and the audio stream read **one shared clock**. On the same LTX-2.3 22B recipe used in the paper, mean shot-boundary error falls from **1.11 s → 0.042 s** — about one frame at 24 fps.

This repository is the official **training and inference** code. Checkpoints and the 200 held-out test prompts live on [Hugging Face](https://huggingface.co/starry0929/Separating-What-From-When).

## 🔥 News
- `[2026.08.25]` 🎉 Release of **inference & training code**, LoRA weights, and 200 held-out test prompts.
- `[2026.08.21]` 🔥 Project page with generated examples: [dagroup-pku.github.io/Temporal-Context-Routing.github.io](https://dagroup-pku.github.io/Temporal-Context-Routing.github.io/).

## 📑 Todo List

- [x] Inference code
- [x] Training code (`configs/tcr_av_lora.yaml`, the paper recipe)
- [x] LoRA weights on Hugging Face
- [x] 200 held-out test prompts on Hugging Face
- [ ] Paper / arXiv

## 🎥 Demo

**The video below is a compressed preview. Full-resolution clips with per-shot clocks are best viewed on the [Project Page](https://dagroup-pku.github.io/Temporal-Context-Routing.github.io/).**


[![Watch the project overview](assets/demo_poster.jpg)](https://dagroup-pku.github.io/Temporal-Context-Routing.github.io/assets/videos/tcr_promo_film.mp4)



## ⚙️ Usage

### Requirements

- Linux + CUDA (PyTorch CUDA 12.4+ wheels via pip)
- One 80 GB GPU for 22B + Gemma
- Local copies of **LTX-2.3 22B** (`.safetensors`), **Gemma 3 12B**, and the TCR LoRA

Every script takes `--checkpoint` / `--model-path`, `--text-encoder-path`, and `--lora-path` / `--dataset` / `--data-root` explicitly.

### Install

```bash
git clone https://github.com/DAGroup-PKU/Temporal-Context-Routing.git
cd Temporal-Context-Routing
conda env create -f environment.yml
conda activate tcr
bash setup_env.sh
# optional China mirror:
# PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_env.sh
```

`setup_env.sh` creates `tcr` if missing, installs torch via pip, then `pip install -e` the three workspace packages (`ltx-core`, `ltx-pipelines`, `ltx-trainer`).

```bash
huggingface-cli download starry0929/Separating-What-From-When \
  separating-what-from-when.safetensors \
  --local-dir ./weights

huggingface-cli download starry0929/Separating-What-From-When \
  test_prompts_200.json \
  --local-dir ./examples
```

### Inference

`scripts/infer.sh` only takes model paths. The built-in prompt is `examples/clip_32845.json` (10 s, 241 frames @ 24 fps, 704×1280).

```bash
bash scripts/infer.sh \
  --checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /path/to/gemma-3-12b-it \
  --lora-path ./weights/separating-what-from-when.safetensors \
  --output outputs/tcr_infer.mp4
```

The 200 held-out scripts are `test_prompts_200.json` on Hugging Face. Point `--prompt-file` at any one of them, or at your own script JSON with `time_range` on every shot and line.

### Training

Dataset JSON (see `examples/dataset.json`):

```json
[
  {"caption": "{... script JSON with time_range ...}", "media_path": "clips/001.mp4"}
]
```

```bash
bash scripts/train.sh \
  --model-path /path/to/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /path/to/gemma-3-12b-it \
  --dataset /path/to/dataset.json \
  --output-dir ./outputs/tcr_av_lora
```

If latents are already computed:

```bash
bash scripts/train.sh \
  --model-path /path/to/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /path/to/gemma-3-12b-it \
  --data-root /path/to/preprocessed \
  --output-dir ./outputs/tcr_av_lora
```

`configs/tcr_av_lora.yaml` matches the paper: LoRA rank 128, `temporal_mode: tcr`, `tcr_beta: 5.0`, 704×1280, 24 fps, joint audio–video.

## 🙏 Acknowledgement

Our work builds on [LTX-2](https://github.com/Lightricks/LTX-2) and [Gemma](https://huggingface.co/google/gemma-3-12b-it). Demos, the overview film, and per-clip clocks are hosted on the [project page](https://dagroup-pku.github.io/Temporal-Context-Routing.github.io/).

## ✏️ Citation

If you find this work useful, please consider giving a ⭐ and citing:

```bibtex
@inproceedings{liu2027missingtimeline,
  title     = {The Missing Timeline: Temporal Context Routing
               for Script-Aligned Joint Audio-Video Generation},
  author    = {Liu, Yichen and Zhang, Quanwei and Wang, Haozhe and
               Zhou, Donghao and Li, Xiaojie and Shi, Yang and Liu, Jiaming and
               Huang, Ruihua and Zou, Yingtian and Zhou, Daquan},
  booktitle = {Under review},
  year      = {2027}
}
```

## License

This repository is released under the [LTX-2 Community License](LICENSE). The LoRA is trained on LTX-2.3 and is intended to be used with that backbone under the same terms.
