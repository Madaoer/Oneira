#!/usr/bin/env bash
# Segment 1 walks into a room where a clay pot appears; segment 2 sets that pot on fire.
# Paths and sampling settings are in scripts/inference.sh; the result goes to results/explore/.
set -euo pipefail

bash scripts/inference.sh --input examples/explore "$@"
