#!/usr/bin/env bash
# Quick end-to-end check that the training pipeline runs without error.
# Trains for 1 epoch on 50 examples — takes minutes, not hours.
# Run this before ANY full training run. If it fails here, the full
# run would fail too, just after wasting real GPU time to find out.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Smoke test: Gemma 3 4B QLoRA, 50 examples, 1 epoch ==="

python3 train.py \
  experiment_name=smoke_test \
  data.max_samples=50 \
  training.num_train_epochs=1 \
  training.eval_steps=0.5 \
  training.save_steps=0.5

echo "=== Smoke test finished — check the output above for errors ==="
