#!/usr/bin/env bash
# LoRA fine-tuning of MiniMax-H3 Ref2VA on the encoded clips.
# The training recipe (LoRA, learning rate, epochs, seed) is in configs/train.yaml.
# Multiple nodes: run this on every node and add
#   --nnodes <N> --node_rank <i> --master_addr <node 0 address> --master_port 29500
# to the torchrun line.
set -euo pipefail

H3_CKPT=checkpoints/MiniMax-H3          # MiniMax-H3, Ref2VA/ inside
LMDB_DIR=data/lmdb                      # encoded clips from scripts/data_process.sh

torchrun --nproc_per_node=8 -m oneira.train \
  --config      configs/train.yaml \
  --base-model  "${H3_CKPT}/Ref2VA" \
  --lmdb        "${LMDB_DIR}" \
  --output      runs/oneira \
  "$@"
