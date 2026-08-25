<div align="center">

# Separating What from When

### Fine-Grained Temporal Control for Joint Audio-Video Generation

**Yichen Liu**<sup>1</sup> · **Quanwei Zhang**<sup>2</sup> · **Haozhe Wang**<sup>3</sup> · **Donghao Zhou**<sup>4</sup> · **Yang Shi**<sup>2</sup> · **Jiaming Liu**<sup>2</sup> · **Ruihua Huang**<sup>2</sup> · **Yingtian Zou**<sup>5</sup> · **Daquan Zhou**<sup>1</sup>

<sup>1</sup>Peking University &nbsp;·&nbsp; <sup>2</sup>Qwen Business Unit of Alibaba &nbsp;·&nbsp; <sup>3</sup>HKUST &nbsp;·&nbsp; <sup>4</sup>CUHK &nbsp;·&nbsp; <sup>5</sup>Shanghai Jiao Tong University

[![Project Page](https://img.shields.io/badge/Project-Page-4d86c4?style=for-the-badge)](https://dagroup-pku.github.io/SEPARATING-WHAT-FROM-WHEN/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Weights-ffbd2e?style=for-the-badge)](https://huggingface.co/starry0929/Separating-What-From-When)
[![GitHub](https://img.shields.io/badge/Code-TCR-24292f?style=for-the-badge&logo=github)](https://github.com/DAGroup-PKU/TCR)

</div>

<p align="center">
  <a href="https://dagroup-pku.github.io/SEPARATING-WHAT-FROM-WHEN/">
    <img src="https://dagroup-pku.github.io/SEPARATING-WHAT-FROM-WHEN/assets/posters/tcr_promo_film_poster.jpg" alt="Separating What from When" width="88%">
  </a>
</p>

**Temporal Context Routing (TCR)** is a training-time, plug-in pathway that separates *what* a shot or a line describes from *when* it should influence generation. Intervals bypass the text encoder and arrive as a timing term on the text-attention logits, so the video stream and the audio stream read one shared clock. On the same LTX-2.3 22B recipe used in the paper, mean shot-boundary error falls from **1.11 s** to **0.042 s** — about one frame at 24 fps.

This repository is the official **training and inference** code. Checkpoints and the 200 held-out test prompts live on [Hugging Face](https://huggingface.co/starry0929/Separating-What-From-When). Playable demos and the overview film are on the [project page](https://dagroup-pku.github.io/SEPARATING-WHAT-FROM-WHEN/).

## News

- `[2026.08.25]` Release of inference & training code, LoRA weights, and 200 held-out test prompts.
- `[2026.08.21]` Project page with generated examples: [dagroup-pku.github.io/SEPARATING-WHAT-FROM-WHEN](https://dagroup-pku.github.io/SEPARATING-WHAT-FROM-WHEN/).

## Todo

- [x] Inference code
- [x] Training code (`configs/tcr_av_lora.yaml`, the paper recipe)
- [x] LoRA weights on Hugging Face
- [x] 200 held-out test prompts on Hugging Face
- [ ] Paper / arXiv

## Requirements

- Linux + CUDA (PyTorch CUDA 12.4+ wheels via pip)
- One 80 GB GPU for 22B + Gemma
- Local copies of:
  - **LTX-2.3 22B** checkpoint (`.safetensors`)
  - **Gemma 3 12B** text-encoder directory
  - **TCR LoRA** — [`separating-what-from-when.safetensors`](https://huggingface.co/starry0929/Separating-What-From-When/blob/main/separating-what-from-when.safetensors)

Every script takes `--checkpoint` / `--model-path`, `--text-encoder-path`, and `--lora-path` / `--dataset` / `--data-root` explicitly.

## Install

```bash
git clone https://github.com/DAGroup-PKU/TCR.git
cd TCR
conda env create -f environment.yml
conda activate tcr
bash setup_env.sh
# optional China mirror:
# PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash setup_env.sh
```

`setup_env.sh` creates `tcr` if missing, installs torch via pip, then `pip install -e` the three workspace packages (`ltx-core`, `ltx-pipelines`, `ltx-trainer`).

Download the released LoRA (and, if you want the evaluation set, the 200 test prompts):

```bash
huggingface-cli download starry0929/Separating-What-From-When \
  separating-what-from-when.safetensors \
  --local-dir ./weights

huggingface-cli download starry0929/Separating-What-From-When \
  test_prompts_200.json \
  --local-dir ./examples
```

## Inference

`scripts/infer.sh` only takes model paths. The built-in prompt is `examples/clip_32845.json` (10 s, 241 frames @ 24 fps, 704×1280) — the opening demo on the project page.

```bash
bash scripts/infer.sh \
  --checkpoint /path/to/ltx-2.3-22b-dev.safetensors \
  --text-encoder-path /path/to/gemma-3-12b-it \
  --lora-path ./weights/separating-what-from-when.safetensors \
  --output outputs/tcr_infer.mp4
```

The 200 held-out scripts used in the paper are `test_prompts_200.json` on Hugging Face. Point `--prompt-file` at any one of them, or at your own MTSS JSON with `time_range` on every shot and line.

## Training

Dataset JSON (see `examples/dataset.json`):

```json
[
  {"caption": "{... MTSS JSON with time_range ...}", "media_path": "clips/001.mp4"}
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

## Acknowledgement

Our work builds on [LTX-2](https://github.com/Lightricks/LTX-2) and [Gemma](https://huggingface.co/google/gemma-3-12b-it). Demos, the overview film, and per-clip clocks are hosted on the [project page](https://dagroup-pku.github.io/SEPARATING-WHAT-FROM-WHEN/).

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{liu2027separating,
  title     = {Separating What from When: Fine-Grained Temporal Control
               for Joint Audio-Video Generation},
  author    = {Liu, Yichen and Zhang, Quanwei and Wang, Haozhe and
               Zhou, Donghao and Shi, Yang and Liu, Jiaming and
               Huang, Ruihua and Zou, Yingtian and Zhou, Daquan},
  booktitle = {Under review},
  year      = {2027}
}
```

## License

This repository is released under the [LTX-2 Community License](LICENSE). The LoRA is trained on LTX-2.3 and is intended to be used with that backbone under the same terms.
