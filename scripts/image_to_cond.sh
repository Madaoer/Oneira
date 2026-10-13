#!/usr/bin/env bash
# Build a sample folder (first_frame.png, cond.mp4, prompt.json) from one image and a spec; see Image to Condition Video in README.md.
#   bash scripts/image_to_cond.sh --image room.png --spec configs/image_to_cond/interaction.yaml --output my_samples/room
# The models are downloaded by `bash scripts/download.sh --image-to-cond`.
set -euo pipefail

LINGBOT_MAP_CKPT=checkpoints/LingBot-Map/lingbot-map.pt
GEOCALIB_CKPT=checkpoints/GeoCalib/geocalib-pinhole.tar
SAM3_CKPT=checkpoints/SAM3
MOGE_CKPT=checkpoints/MoGe-2/model.pt

python -m oneira.image_to_cond \
  --image        examples/memory/segment_1/first_frame.png \
  --spec         configs/image_to_cond/interaction.yaml \
  --output       my_samples/interaction \
  --lingbot-map  "${LINGBOT_MAP_CKPT}" \
  --geocalib     "${GEOCALIB_CKPT}" \
  --sam3         "${SAM3_CKPT}" \
  --moge         "${MOGE_CKPT}" \
  "$@"
