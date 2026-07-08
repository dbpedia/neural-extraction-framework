#!/usr/bin/env bash
# Experiment 1, learning rate 2e-4. Requires prepare_data.py to have
# been run first.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Experiment 1 — learning rate 2e-4 ==="
python3 train.py training.learning_rate=2e-4
echo "=== Run finished ==="
