#!/usr/bin/env bash
# A wicker basket is set on fire and the camera turns away; segment 2 turns back with two memory frames
# and the basket is still burning.
# Paths and sampling settings are in scripts/inference.sh; the result goes to results/memory/.
set -euo pipefail

bash scripts/inference.sh --input examples/memory "$@"
