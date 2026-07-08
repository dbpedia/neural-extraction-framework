#!/usr/bin/env bash
# Experiment 1, learning rate 1e-5. Requires prepare_data.py to have
# been run first.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Experiment 1 — learning rate 1e-5 ==="
python3 train.py training.learning_rate=1e-5
echo "=== Run finished ==="
