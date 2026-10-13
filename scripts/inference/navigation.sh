#!/usr/bin/env bash
# Walking along a sidewalk while a parked car drives off; in the second half the camera steps left and turns slightly right. One segment.
# Paths and sampling settings are in scripts/inference.sh; the result goes to results/navigation/.
set -euo pipefail

bash scripts/inference.sh --input examples/navigation "$@"
