#!/usr/bin/env bash
# A plush cow toy that is not in the training data is made invisible. One segment.
# Paths and sampling settings are in scripts/inference.sh; the result goes to results/single_action/.
set -euo pipefail

bash scripts/inference.sh --input examples/single_action "$@"
