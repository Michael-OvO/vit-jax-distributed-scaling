#!/bin/bash
# Run on GCP TPU VM (v2-8, v3-8, or v4-8 single-host).
# TPU MXUs are native bf16, so every experiment here sets --precision bf16.
set -euo pipefail
source venv/bin/activate 2>/dev/null || true

# Silence TensorFlow's CUDA / GPU chatter on TPU hosts; only tfds uses it and
# we pinned tensorflow-cpu, but the warnings are still noisy.
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== GCP TPU Training ==="
echo "Detected devices:"
python -c "import jax; print(f'  {len(jax.devices())} devices: {jax.devices()}')"

# Training with bf16 precision (native on TPU)
echo "--- Running training ---"
python main.py \
    --experiment train \
    --batch_size 1024 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --precision bf16 \
    --output_dir ./outputs/tpu_train \
    "$@"

# Scaling experiment
echo "--- Running scaling experiment ---"
python main.py \
    --experiment scaling \
    --batch_size 1024 \
    --precision bf16 \
    --output_dir ./outputs/tpu_scaling \
    "$@"

# Straggler experiment
echo "--- Running straggler experiment ---"
python main.py \
    --experiment straggler \
    --batch_size 1024 \
    --precision bf16 \
    --straggler_delay 1000 \
    --output_dir ./outputs/tpu_straggler \
    "$@"

echo "=== All experiments complete. Results in ./outputs/ ==="
