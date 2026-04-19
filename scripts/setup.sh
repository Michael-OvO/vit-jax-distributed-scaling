#!/bin/bash
# Environment setup for ViT-JAX Distributed training.
#
# Detects the accelerator at the *host* level (TPU / NVIDIA GPU / CPU) and
# installs the matching JAX build exactly once. The rest of requirements.txt
# is intentionally JAX-free, so it can be installed on top of any JAX backend
# without clobbering it.
set -euo pipefail

echo "=== ViT-JAX Distributed Training Setup ==="

# ---- 1. Detect platform ----------------------------------------------------
# TPU VMs expose /dev/accel* and the `tpu` line in GCE metadata. We check both
# device files and env vars — no reliance on libtpu being importable, which
# requires JAX to already be installed.
PLATFORM="cpu"
if ls /dev/accel* >/dev/null 2>&1 \
    || [ -d /sys/class/accel ] \
    || [ "${TPU_NAME:-}" != "" ] \
    || [ "${COLAB_TPU_ADDR:-}" != "" ]; then
    PLATFORM="tpu"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    PLATFORM="gpu"
fi
echo "Detected platform: ${PLATFORM}"

# ---- 2. Choose pip target --------------------------------------------------
# On GCP TPU VMs the box is single-tenant and throwaway, and the base image
# ships *without* python3-venv. Follow Google's official TPU guidance and
# install to the user site-packages (pip --user) instead of a venv.
# On GPU/CPU hosts (usually developer laptops or shared VMs) we still isolate
# with a venv to avoid polluting the system Python.
if [ "${PLATFORM}" = "tpu" ]; then
    PIP=(python3 -m pip install --user)
    echo "Using pip --user (no venv on TPU VM)."
    export PATH="$HOME/.local/bin:${PATH}"
    PY=python3
else
    if [ ! -d venv ]; then
        python3 -m venv venv
    fi
    # shellcheck disable=SC1091
    source venv/bin/activate
    PIP=(pip install)
    PY=python
fi
"${PIP[@]}" --upgrade pip wheel

# ---- 3. Install JAX for the detected backend -------------------------------
case "${PLATFORM}" in
    tpu)
        echo "Installing JAX for TPU..."
        "${PIP[@]}" -U "jax[tpu]>=0.4.20" \
            -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
        ;;
    gpu)
        echo "Installing JAX for NVIDIA GPU (CUDA 12)..."
        "${PIP[@]}" -U "jax[cuda12]>=0.4.20"
        ;;
    cpu)
        echo "Installing CPU-only JAX..."
        "${PIP[@]}" -U "jax>=0.4.20"
        ;;
esac

# ---- 4. Install the rest of the dependencies (no JAX here) -----------------
"${PIP[@]}" -r requirements.txt

# ---- 5. Pre-download CIFAR-100 ---------------------------------------------
echo "Pre-downloading CIFAR-100 dataset..."
$PY -c "import tensorflow_datasets as tfds; tfds.load('cifar100', split='train', batch_size=1)"

# ---- 6. Confirm device visibility ------------------------------------------
$PY - <<'PY'
import jax
print(f"JAX version:  {jax.__version__}")
print(f"Backend:      {jax.default_backend()}")
print(f"Devices ({jax.local_device_count()}): {jax.devices()}")
PY

echo "=== Setup complete ==="
