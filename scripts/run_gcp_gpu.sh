#!/bin/bash
# Run on GCP multi-GPU VM
# Recommended: n1-standard-8 with 4x NVIDIA T4 or A100
set -e
source venv/bin/activate 2>/dev/null || true

echo "=== GCP Multi-GPU Training ==="
echo "Detected devices:"
python3 -c "import jax; print(f'  {len(jax.devices())} devices: {jax.devices()}')"

# Full training run
echo "--- Running training ---"
python3 main.py \
    --experiment train \
    --batch_size 512 \
    --epochs 20 \
    --learning_rate 1e-3 \
    --output_dir ./outputs/gpu_train \
    "$@"

# Scaling experiment
echo "--- Running scaling experiment ---"
python3 main.py \
    --experiment scaling \
    --batch_size 512 \
    --output_dir ./outputs/gpu_scaling \
    "$@"

# Straggler experiment
echo "--- Running straggler experiment ---"
python3 main.py \
    --experiment straggler \
    --batch_size 512 \
    --straggler_delay 1000 \
    --output_dir ./outputs/gpu_straggler \
    "$@"

echo "=== All experiments complete. Results in ./outputs/ ==="
