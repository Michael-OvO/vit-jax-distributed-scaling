#!/bin/bash
# Run training locally (CPU or single GPU)
set -e
source venv/bin/activate 2>/dev/null || true

echo "=== Local Training Run ==="
python3 main.py \
    --experiment train \
    --batch_size 128 \
    --epochs 10 \
    --learning_rate 1e-3 \
    --output_dir ./outputs/local_train \
    "$@"
