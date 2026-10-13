#!/usr/bin/env bash
# Generate a video for each sample; the result goes to <output>/<sample>/generated.mp4.
# scripts/inference/ has one script per example; for your own samples, e.g.
#   bash scripts/inference.sh --input my_samples/scene_01
# A sample with segment_1/, segment_2/, ... is generated segment by segment; --input examples/* runs all of them.
# Sampling settings that rarely change are in configs/inference.yaml.
set -euo pipefail

H3_CKPT=checkpoints/MiniMax-H3          # MiniMax-H3, Ref2VA/ inside
ONEIRA_CKPT=checkpoints/Oneira          # Oneira LoRA, model.safetensors

python -m oneira.inference \
  --config      configs/inference.yaml \
  --base-model  "${H3_CKPT}/Ref2VA" \
  --checkpoint  "${ONEIRA_CKPT}/model.safetensors" \
  --input       examples/multi_instance \
  --output      results \
  --num-steps   20 \
  --seed        128 \
  "$@"
