#!/usr/bin/env bash
# Encode the dataset into LMDB, one shard per GPU. Clips already in the LMDB are skipped.
set -euo pipefail

H3_CKPT=checkpoints/MiniMax-H3          # MiniMax-H3, Ref2VA/ inside
DATA_DIR=data/Oneira_example_data       # downloaded training data
LMDB_DIR=data/lmdb                      # encoded clips

torchrun --standalone --nproc_per_node=8 -m oneira.data_process \
  --base-model   "${H3_CKPT}/Ref2VA" \
  --data-root    "${DATA_DIR}" \
  --output       "${LMDB_DIR}" \
  --num-workers  4 \
  "$@"
