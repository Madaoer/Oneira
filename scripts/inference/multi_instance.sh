#!/usr/bin/env bash
# Five identical plaster busts; only the fourth from the left turns into gold. One segment.
# Paths and sampling settings are in scripts/inference.sh; the result goes to results/multi_instance/.
set -euo pipefail

bash scripts/inference.sh --input examples/multi_instance "$@"
