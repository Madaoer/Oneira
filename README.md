<h1 align="center">
  <img src="assets/icon.svg" height="60" align="center" alt="" />&nbsp;Oneira
</h1>

<p align="center"><b>From Open-Ended Generation to Open-World Interaction in Video World Models</b></p>

<div align="center">

[![Website](https://img.shields.io/badge/Website-Oneira-blue)](https://madaoer.github.io/projects/oneira/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://madaoer.github.io/projects/oneira/assets/oneira.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2610.01614-red)](https://arxiv.org/abs/2610.01614)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Oneira-yellow)](https://huggingface.co/Madaoer-Otaku/Oneira_example_weights)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Oneira-yellow)](https://huggingface.co/datasets/Madaoer-Otaku/Oneira_example_data)

</div>

> #### [Oneira: From Open-Ended Generation to Open-World Interaction in Video World Models](https://arxiv.org/abs/2610.01614)
>
> ##### [Xindi Yang](https://madaoer.github.io/), [Baolu Li](https://libaolu312.github.io/), [Liam Lee](https://madaoer.github.io/projects/oneira/#), [Zhenfei Yin](https://yinzhenfei.github.io/), [Songxin Zhang](https://madaoer.github.io/projects/oneira/#), [Zhuoyang Song](https://madaoer.github.io/projects/oneira/#), [Xu Jia](https://stephenjia.github.io/), [Jianfei Cai](https://jianfei-cai.github.io/), [Tien-Tsin Wong](https://ttwong12.github.io/), [Bingyi Jing](https://scholar.google.com/citations?user=k83Zz2MAAAAJ&hl=en), [Mengyue Yang](https://ymy4323460.github.io/)&dagger;  (&dagger; corresponding author)

<p align="center">
  <img src="assets/teaser.png" width="100%" alt="Oneira teaser" />
</p>

Oneira is an interactive video world model that closes the loop between generation and interaction through an explicit world state managed by a coding agent. **What appears in the generated world becomes actionable**: when exploration reveals a new object, the agent adds it to the world state, so it can be acted on in later segments. **What the agent changes remains changed**: interaction outcomes are written back into the world state and carried across segments. The world state is rendered into a coarse condition video, and the video model fills in the appearance, motion and interaction details.

## 📢 News

- [2026/10/13] We release the code, the dataset and the model weights.
- [2026/10/01] We release the Oneira paper.

## 🛠️ Setup

### Requirements

- Linux with NVIDIA GPUs of 80 GB memory (A100, H100 or newer)
- CUDA 12.8 and Python 3.12

### Installation

```bash
git clone https://github.com/Madaoer/Oneira.git
cd Oneira
conda create -n oneira python=3.12 -y
conda activate oneira
pip install -r requirements.txt
pip install ninja
MAX_JOBS=32 pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
```

### Checkpoints

| Weights | Source | Files |
| --- | --- | --- |
| MiniMax-H3 Ref2VA | [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/Ref2VA) | DiT, VAE, text encoder under `Ref2VA/` |
| Oneira LoRA | [`Madaoer-Otaku/Oneira_example_weights`](https://huggingface.co/Madaoer-Otaku/Oneira_example_weights) | `model.safetensors` |

`scripts/download.sh` downloads everything by default, or only the parts given as options:

```bash
bash scripts/download.sh                             # everything
bash scripts/download.sh --checkpoints --examples    # only what inference needs
```

| Option | Downloads |
| --- | --- |
| `--checkpoints` | MiniMax-H3 Ref2VA and the Oneira LoRA, into `checkpoints/` |
| `--examples` | the [examples](#examples), into `examples/` |
| `--data` | the training data, into `data/` |
| `--image-to-cond` | the models for Image to Condition Video, into `checkpoints/` (SAM3 needs [access approval](https://huggingface.co/facebook/sam3) first) |

The scripts in `scripts/` set these paths at the top (`H3_CKPT`, `ONEIRA_CKPT`, `DATA_DIR`, `LMDB_DIR`); change them there to store the files elsewhere.

## 🗂️ Dataset

[`Oneira_example_data`](https://huggingface.co/datasets/Madaoer-Otaku/Oneira_example_data) has 5k clips: 2.5k navigation and 2.5k interaction. Every clip is 243 frames at 24 fps, 832x480, split into 8 chunks of about 1.27 s. The target video and the condition video have the same size.

| Split | Clips | Content |
| --- | --- | --- |
| `navigation` | 2.5k | the camera moves through a scene and nothing is touched |
| `interaction` | 2.5k | a hand acts on an object: pick up, open, close, put out a fire with an extinguisher, set on fire, freeze, put out a fire with ice, two-step magic, make invisible, gild, smash, shoot |

Download with `bash scripts/download.sh --data`.

### Structure

<details>
<summary><b>Data Structure</b> (click to expand)</summary>

```
Oneira_example_data/
├── navigation/
│   └── <clip_id>/
│       ├── video.mp4          # target video
│       ├── cond.mp4           # condition video
│       └── prompt.json
├── interaction/
│   └── <clip_id>/
│       ├── video.mp4
│       ├── cond.mp4
│       ├── prompt.json
│       └── memory_*.png       # optional memory frames (1-2)
└── examples/                  # inference examples, see Examples
```

</details>

### Condition Video

The condition video is rendered from the world state table. It shows ground grid lines, a wireframe lattice and one 3D box per object. The box colour encodes the object state:

| Colour | State |
| --- | --- |
| purple | default state |
| yellow | first non-default state |
| blue | second non-default state |

### Memory Frames

Some clips include 1-2 memory frames of the same place, which the model takes as additional reference images.

## 🚀 Inference

Given a first frame, a condition video and a caption for each chunk, Oneira generates a 243-frame video at 24 fps. The model is a LoRA on [MiniMax-H3 Ref2VA](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/main/Ref2VA); `scripts/download.sh` downloads both (see [Checkpoints](#checkpoints)).

### Sample Format

One folder per sample:

```
<sample>/
├── first_frame.png
├── cond.mp4            # condition video, 243 frames at 24 fps
├── prompt.json         # global caption + 8 chunk captions
└── memory_*.png        # optional, 1-2 memory frames
```

A longer video is a chain of segments. Only the first segment has a first frame; each later segment starts from the last frame generated for the one before it:

```
<sample>/
├── segment_1/
│   ├── first_frame.png
│   ├── cond.mp4
│   └── prompt.json
└── segment_2/
    ├── cond.mp4
    ├── prompt.json
    └── memory_*.png    # optional
```

The 8 chunks are about 1.27 s each. A chunk without interaction reads `Nothing happens.`; an interaction is written as the action followed by its outcome:

```json
{
  "global_caption": "...",
  "chunks": [
    {"k": 1, "caption": "Nothing happens."},
    ...
    {"k": 5, "caption": "A hand picks up the red apple. The red apple is lifted off the table and carried."},
    ...
  ]
}
```

### Examples

| Sample | Content |
| --- | --- |
| `navigation` | walk along a sidewalk while a parked car drives off |
| `single_action` | a plush cow toy unseen in training is made invisible |
| `multi_instance` | of five identical busts, only the fourth from the left turns gold |
| `explore` | a clay pot that appears in segment 1 is set on fire in segment 2 |
| `memory` | a basket set on fire is still burning when the camera turns back in segment 2 |

```bash
bash scripts/inference/multi_instance.sh                # scripts/inference/ has one script per example
bash scripts/inference.sh --input my_samples/scene_01   # your own sample
```

`scripts/inference.sh` sets the checkpoints, input, output and sampling settings; arguments added after a script override them, e.g. `bash scripts/inference/single_action.sh --seed 128`.

## 🖼️ Image to Condition Video

`oneira.image_to_cond` makes a condition video from a single photo: it estimates the 3D layout of the photo, places boxes on the objects listed in the spec, plans the camera motion and renders the condition video. The output folder can be used directly for inference.

### Extra Dependencies

On top of the [installation](#installation) above:

```bash
pip install --no-deps -r requirements_image_to_cond.txt
conda install -c conda-forge libegl-devel libgl-devel -y   # only if the system has no libEGL.so / libGL.so
bash scripts/download.sh --image-to-cond                    # the four models below, into checkpoints/
```

| Model | Source | Used for |
| --- | --- | --- |
| LingBot-Map | [`robbyant/lingbot-map`](https://huggingface.co/robbyant/lingbot-map) | point map and intrinsics |
| GeoCalib | [`cvg/GeoCalib`](https://github.com/cvg/GeoCalib) | gravity direction |
| SAM3 | [`facebook/sam3`](https://huggingface.co/facebook/sam3) | ground and object masks |
| MoGe-2 | [`Ruicheng/moge-2-vitl-normal`](https://huggingface.co/Ruicheng/moge-2-vitl-normal) | metric depth for the camera height |

### Usage

```bash
bash scripts/image_to_cond.sh        # walks up to the wicker basket in examples/memory and sets it on fire, into my_samples/interaction
bash scripts/image_to_cond.sh \
  --image  examples/navigation/first_frame.png \
  --spec   configs/image_to_cond/navigation.yaml \
  --output my_samples/navigation     # walks along the sidewalk while the parked car drives off
bash scripts/inference.sh --input my_samples/navigation
```

For your own photo, copy a spec from `configs/image_to_cond/` and edit it. The output folder holds `first_frame.png`, `cond.mp4` and `prompt.json`. Perception results are cached in `<output>/perception/`, so rerunning after editing the spec is fast.

## 🏋️ Training

Oneira is a LoRA fine-tuned from MiniMax-H3 Ref2VA. Training has two stages: the clips are first encoded into latents once, and the LoRA is then trained on the cached latents.

### Data Processing

`scripts/data_process.sh` runs the MiniMax-H3 VAE and text encoder over every clip and writes the latents to LMDB, one shard per GPU. The target video and the first frame are encoded at full resolution and the condition video at 448x256.

```bash
bash scripts/download.sh --checkpoints --data
bash scripts/data_process.sh
```

Any data organised like `Oneira_example_data` (see [Structure](#structure)) can be processed the same way.

### Fine-tuning

```bash
bash scripts/train.sh
```

`scripts/train.sh` sets the paths; the training recipe (LoRA, learning rate, epochs, seed) is in `configs/train.yaml`. Arguments added after the script override both, e.g. `bash scripts/train.sh --lr 5e-5`. For multiple nodes, run the script on every node with the extra `torchrun` arguments noted at the top of the script. The LoRA weights are written to `runs/oneira/epoch-N.safetensors` at the end of each epoch.

## 📋 TODO

- [x] Inference code and model weights
- [x] Dataset: 5k training clips (2.5k navigation + 2.5k interaction)
- [x] Training code
- [ ] Coding agent harness

## 📖 Citation

```bibtex
@article{yang2026oneira,
  title={Oneira: From Open-Ended Generation to Open-World Interaction in Video World Models},
  author={Yang, Xindi and Li, Baolu and Lee, Liam and Yin, Zhenfei and Zhang, Songxin and Song, Zhuoyang and Jia, Xu and Cai, Jianfei and Wong, Tien-Tsin and Jing, Bingyi and Yang, Mengyue},
  journal={arXiv preprint arXiv:2610.01614},
  year={2026}
}
```

## ✨ Acknowledgement

Oneira is fine-tuned from [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) Ref2VA, and our codebase is adapted from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). Thanks for their great work!

## 📜 License

This project is licensed under the [MIT License](LICENSE).
