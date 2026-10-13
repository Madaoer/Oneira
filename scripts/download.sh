#!/usr/bin/env bash
# Download what Oneira needs: everything by default, or only the parts given as options:
#   --checkpoints     MiniMax-H3 Ref2VA and the Oneira LoRA
#   --examples        the inference examples
#   --data            the training data
#   --image-to-cond   the models for Image to Condition Video (SAM3 needs access approval on Hugging Face)
# e.g. `bash scripts/download.sh --checkpoints --examples` for inference only.
# Change the paths below to store them elsewhere.
set -euo pipefail

H3_CKPT=checkpoints/MiniMax-H3          # MiniMax-H3, Ref2VA/ inside
ONEIRA_CKPT=checkpoints/Oneira          # Oneira LoRA, model.safetensors
EXAMPLE_ROOT=.                          # examples/ goes inside
DATA_DIR=data/Oneira_example_data       # training data
LINGBOT_MAP_DIR=checkpoints/LingBot-Map # lingbot-map.pt
GEOCALIB_DIR=checkpoints/GeoCalib       # geocalib-pinhole.tar
SAM3_DIR=checkpoints/SAM3               # gated: request access on Hugging Face and run `hf auth login`
MOGE_DIR=checkpoints/MoGe-2             # model.pt

CHECKPOINTS=0
EXAMPLES=0
DATA=0
IMAGE_TO_COND=0
if [[ $# == 0 ]]; then
  CHECKPOINTS=1; EXAMPLES=1; DATA=1; IMAGE_TO_COND=1
fi
for arg in "$@"; do
  case "${arg}" in
    --checkpoints)   CHECKPOINTS=1 ;;
    --examples)      EXAMPLES=1 ;;
    --data)          DATA=1 ;;
    --image-to-cond) IMAGE_TO_COND=1 ;;
    *)               echo "unknown option: ${arg}" >&2; exit 1 ;;
  esac
done

if [[ "${CHECKPOINTS}" == 1 ]]; then
  hf download MiniMaxAI/MiniMax-H3 \
    --include    "Ref2VA/*" \
    --local-dir  "${H3_CKPT}"

  hf download Madaoer-Otaku/Oneira_example_weights \
    --local-dir  "${ONEIRA_CKPT}"
fi

if [[ "${EXAMPLES}" == 1 ]]; then
  hf download Madaoer-Otaku/Oneira_example_data \
    --repo-type  dataset \
    --include    "examples/*" \
    --local-dir  "${EXAMPLE_ROOT}"
fi

if [[ "${DATA}" == 1 ]]; then
  hf download Madaoer-Otaku/Oneira_example_data \
    --repo-type  dataset \
    --local-dir  "${DATA_DIR}"
fi

if [[ "${IMAGE_TO_COND}" == 1 ]]; then
  hf download robbyant/lingbot-map lingbot-map.pt \
    --local-dir  "${LINGBOT_MAP_DIR}"

  mkdir -p "${GEOCALIB_DIR}"
  curl -L --fail -o "${GEOCALIB_DIR}/geocalib-pinhole.tar" \
    https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar

  hf download facebook/sam3 \
    --exclude    "sam3.pt" \
    --local-dir  "${SAM3_DIR}"

  hf download Ruicheng/moge-2-vitl-normal model.pt \
    --local-dir  "${MOGE_DIR}"
fi
