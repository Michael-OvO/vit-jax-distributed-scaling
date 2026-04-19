# ViT-JAX Distributed Scaling

Distributed data-parallel training of Vision Transformer (ViT-Small) on CIFAR-100 using JAX/Flax, with structured experiments for **scaling analysis** and **straggler simulation**.

This is a systems-focused ML project: the goal is to understand distributed training behavior — synchronization overhead, throughput scaling, and the impact of slow workers — not just model accuracy.

## Architecture

```
vit_jax_distributed/
├── models/          # ViT-Small (Flax Linen)
├── data/            # CIFAR-100 pipeline (TFDS → numpy)
├── train/           # Training loop with metrics
├── distributed/     # pmap, all-reduce, state replication
├── experiments/     # Scaling + straggler benchmarks
└── utils/           # Logging, timing, CLI config
```

### Model

**ViT-Small** configured for 32×32 CIFAR images:
- Patch size 4 → 64 patches per image
- Hidden dim 384, 6 attention heads, 8 transformer blocks
- ~6M parameters
- Pre-norm architecture (LayerNorm before attention/MLP)

### Distributed Training Approach

**Synchronous data parallelism** using `jax.pmap`:

1. **Replicate** model parameters across all devices
2. **Shard** each batch: `(B, H, W, C)` → `(num_devices, B/N, H, W, C)`
3. **Compute** gradients independently on each device
4. **All-reduce** gradients via `jax.lax.pmean(grads, axis_name='batch')`
5. **Update** parameters identically on every device

This is the standard data-parallel pattern. The `pmean` call triggers an all-reduce collective, which is the key synchronization point. Every device must reach this barrier before any can proceed — this is what makes stragglers costly.

```
Device 0: forward → backward → [pmean] → optimizer step
Device 1: forward → backward → [pmean] → optimizer step
Device 2: forward → backward → [pmean] → optimizer step
Device 3: forward → backward → [pmean] → optimizer step
                                  ↑
                          all-reduce barrier
```

### Experiments

**Scaling experiment**: Measures throughput (images/sec) and step time across device counts. Computes scaling efficiency relative to single-device baseline. Generates throughput, step time, and efficiency plots.

**Straggler experiment**: Injects artificial compute delay on device 0 using `jax.lax.fori_loop` with dummy matrix multiplications, gated by `jax.lax.cond` on `axis_index`. Tests multiple delay magnitudes and measures the global slowdown factor.

## Quick Start

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/vit-jax-distributed-scaling.git
cd vit-jax-distributed-scaling
bash scripts/setup.sh
source venv/bin/activate
```

### Run Training

```bash
# Basic training (auto-detects devices)
python main.py --experiment train --batch_size 512 --epochs 20

# Quick test run
python main.py --experiment train --batch_size 128 --epochs 2 --output_dir ./outputs/test
```

### Run Experiments

```bash
# Scaling benchmark
python main.py --experiment scaling --batch_size 512 --output_dir ./outputs/scaling

# Straggler simulation
python main.py --experiment straggler --batch_size 512 --output_dir ./outputs/straggler
```

## Running on GCP

### Multi-GPU VM

**Recommended**: `n1-standard-8` or `a2-highgpu-4g` with 4× NVIDIA T4/A100.

```bash
# SSH into VM
gcloud compute ssh YOUR_VM_NAME --zone YOUR_ZONE

# Clone and setup
git clone https://github.com/YOUR_USERNAME/vit-jax-distributed-scaling.git
cd vit-jax-distributed-scaling
bash scripts/setup.sh
source venv/bin/activate

# Verify GPU detection
python -c "import jax; print(jax.devices())"

# Run all experiments
bash scripts/run_gcp_gpu.sh
```

The setup script auto-detects NVIDIA GPUs and installs `jax[cuda12]`. JAX sees all GPUs on the machine automatically — no `CUDA_VISIBLE_DEVICES` configuration needed.

### TPU VM

**Recommended**: `v2-8` or `v3-8` (8 TPU cores).

```bash
# Create TPU VM
gcloud compute tpus tpu-vm create vit-training \
    --zone=us-central1-b \
    --accelerator-type=v3-8 \
    --version=tpu-ubuntu2204-base

# SSH in
gcloud compute tpus tpu-vm ssh vit-training --zone=us-central1-b

# Clone and setup
git clone https://github.com/YOUR_USERNAME/vit-jax-distributed-scaling.git
cd vit-jax-distributed-scaling
bash scripts/setup.sh
source venv/bin/activate

# Verify TPU detection
python -c "import jax; print(jax.devices())"
# Should show 8 TpuDevice entries

# Run all experiments with bf16 (native TPU precision)
bash scripts/run_gcp_tpu.sh
```

The setup script detects the TPU runtime (via `/dev/accel*` or GCE TPU env vars — no prior JAX needed) and installs `jax[tpu]` from Google's release bucket. `requirements.txt` itself pins no JAX extra, so it will not clobber the TPU build when it runs next.

`--precision bf16` activates via `jax.config.update("jax_default_matmul_precision", "bfloat16")`: every matmul/einsum (Attention QKV, MLP, Dense head) runs in bf16 on the MXU while parameters and optimizer state stay in fp32. Expect ~2× step throughput versus fp32 on v2/v3 TPUs.

### Key CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment` | `train` | `train`, `scaling`, or `straggler` |
| `--batch_size` | `512` | Global batch size (split across devices) |
| `--epochs` | `20` | Training epochs |
| `--num_devices` | `0` | Device count (0 = all available) |
| `--precision` | `fp32` | `fp32` or `bf16` |
| `--straggler_delay` | `1000` | Dummy compute iterations for straggler |
| `--output_dir` | `./outputs` | Where to save logs and plots |
| `--learning_rate` | `1e-3` | Peak LR (with cosine warmup schedule) |
| `--num_layers` | `8` | Transformer depth |
| `--log_every` | `50` | Steps between log entries |
| `--eval_every` | `1` | Epochs between test evaluation |

See `configs/default.yaml` for all defaults.

## Output

Training produces:
- `step_metrics.csv` — per-step loss, accuracy, timing
- `epoch_metrics.csv` — per-epoch train/test metrics
- `training_log.json` — complete run metadata

Experiments produce:
- `scaling_results.csv/json` + `throughput_vs_devices.png`, `step_time_vs_devices.png`, `scaling_efficiency_vs_devices.png`
- `straggler_results.csv/json` + `step_time_comparison.png`, `slowdown_vs_delay.png`

## Expected Results

### Scaling

On a 4-GPU machine, expect:
- Near-linear throughput scaling from 1→4 devices
- Slight efficiency drop at higher device counts due to all-reduce overhead
- Step time roughly constant (batch is split, not replicated)

### Straggler Impact

With synchronous data parallelism:
- A single slow device forces **all** devices to wait at the all-reduce barrier
- Slowdown is proportional to the straggler's delay
- This demonstrates why async-SGD and straggler mitigation matter in production

## Project Structure

```
.
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── configs/
│   └── default.yaml                 # Documented defaults
├── scripts/
│   ├── setup.sh                     # Environment setup
│   ├── run_local.sh                 # Local run
│   ├── run_gcp_gpu.sh               # Multi-GPU run
│   └── run_gcp_tpu.sh               # TPU run
└── vit_jax_distributed/
    ├── models/vit.py                # ViT-Small implementation
    ├── data/cifar100.py             # CIFAR-100 data pipeline
    ├── distributed/parallel.py      # pmap + all-reduce
    ├── train/trainer.py             # Training loop
    ├── experiments/
    │   ├── scaling.py               # Scaling benchmark
    │   └── straggler.py             # Straggler simulation
    └── utils/
        ├── config.py                # CLI configuration
        ├── logging.py               # Metrics logger
        └── timing.py                # Wall-clock timing
```

## Tech Stack

- **JAX** — XLA-compiled numerical computing with auto-differentiation
- **Flax Linen** — Neural network modules
- **Optax** — Optimizers (AdamW + cosine warmup schedule)
- **TFDS** — Dataset loading (CIFAR-100)
- **matplotlib** — Experiment plots

## Key Design Decisions

1. **pmap over pjit**: For pure data parallelism, `pmap` is simpler and more explicit. `pjit` with sharding constraints is better for model parallelism, which isn't needed here.

2. **numpy-only data pipeline**: After the initial TFDS load, all augmentation and batching uses numpy. This avoids TF runtime overhead and keeps the data path simple.

3. **Straggler via XLA compute**: Rather than `time.sleep()` (which doesn't work inside `pmap`/XLA), we inject real matrix multiplications via `lax.fori_loop` + `lax.cond`. This creates genuine compute imbalance that the XLA scheduler can't optimize away.

4. **Separate experiment configs**: Scaling and straggler experiments use dedicated dataclasses rather than the training config, keeping concerns separate while sharing the same model/data infrastructure.
