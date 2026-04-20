"""Full-test-set evaluation of a trained ViT-Small checkpoint on CIFAR-100.

Runs pmap'd inference over all 10,000 test images, then saves:

  * ``test_predictions.npz`` — per-example predictions for downstream plotting
      - true_labels   : (10000,) int
      - pred_labels   : (10000,) int   (argmax)
      - top5_indices  : (10000, 5) int
      - top5_probs    : (10000, 5) float
      - per_class_acc : (100,) float
      - confusion     : (100, 100) int  (rows=true, cols=predicted)
  * ``test_summary.json`` — scalar metrics (top-1, top-5 accuracy)

Run on a TPU VM:
    python3 scripts/full_test_eval.py \
        --checkpoint ~/outputs/trained_100/checkpoint_latest.msgpack \
        --output_dir ~/outputs/trained_100 \
        --precision bf16
"""

from __future__ import annotations

import argparse
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils

from vit_jax_distributed.data.cifar100 import _load_split
from vit_jax_distributed.distributed.parallel import create_train_state
from vit_jax_distributed.models.vit import ViTSmall
from vit_jax_distributed.utils.checkpoint import load_checkpoint


def _load_model_from_checkpoint(checkpoint_path: str):
    """Rebuild the model from checkpoint metadata and restore parameters."""
    meta_path = os.path.join(
        os.path.dirname(os.path.abspath(checkpoint_path)),
        "checkpoint_latest.json",
    )
    with open(meta_path) as f:
        meta = json.load(f)
    cfg = meta["config"]

    model = ViTSmall(
        num_classes=100,
        patch_size=cfg["patch_size"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        mlp_dim=cfg["mlp_dim"],
        dropout_rate=0.0,
        image_size=cfg["image_size"],
    )
    template = create_train_state(
        jax.random.PRNGKey(0), model, learning_rate=1e-3,
        image_size=cfg["image_size"], total_steps=2,
    )
    state = load_checkpoint(checkpoint_path, template)
    return model, state, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Global batch (must divide 10000 and num_devices)")
    args = parser.parse_args()

    if args.precision == "bf16":
        jax.config.update("jax_default_matmul_precision", "bfloat16")

    os.makedirs(args.output_dir, exist_ok=True)
    num_devices = jax.local_device_count()
    assert args.batch_size % num_devices == 0, \
        f"--batch_size {args.batch_size} not divisible by {num_devices} devices"
    per_device = args.batch_size // num_devices
    print(f"Devices: {num_devices}  per-device batch: {per_device}")

    # ---- Load test set ------------------------------------------------
    print("Loading CIFAR-100 test split...")
    images, labels = _load_split("test")  # (10000, 32, 32, 3) already standardised
    assert images.shape[0] == 10000

    # ---- Load model + checkpoint --------------------------------------
    print(f"Loading checkpoint from {args.checkpoint}")
    model, state, cfg = _load_model_from_checkpoint(args.checkpoint)
    print(f"Model: hidden={cfg['hidden_dim']}, layers={cfg['num_layers']}")

    state = jax_utils.replicate(state)

    # ---- pmap'd predict -----------------------------------------------
    @partial(jax.pmap, axis_name="batch")
    def predict(state, batch):
        logits = state.apply_fn({"params": state.params}, batch, train=False)
        probs = jax.nn.softmax(logits)
        top_probs, top_idx = jax.lax.top_k(probs, 5)
        return top_probs, top_idx

    # ---- Iterate --------------------------------------------------------
    n_batches = images.shape[0] // args.batch_size
    all_top_probs = []
    all_top_idx = []

    for i in range(n_batches):
        chunk = images[i * args.batch_size : (i + 1) * args.batch_size]
        chunk = chunk.reshape((num_devices, per_device) + chunk.shape[1:])
        top_probs, top_idx = predict(state, chunk)
        all_top_probs.append(np.asarray(top_probs).reshape(-1, 5))
        all_top_idx.append(np.asarray(top_idx).reshape(-1, 5))
        if i == 0:
            print(f"  first batch done (includes XLA compile)")
        elif (i + 1) % 5 == 0:
            print(f"  batch {i + 1}/{n_batches}")

    top_probs = np.concatenate(all_top_probs, axis=0)   # (10000, 5)
    top_idx = np.concatenate(all_top_idx, axis=0)       # (10000, 5)
    pred_labels = top_idx[:, 0]                          # argmax
    true_labels = labels[: top_idx.shape[0]]

    # ---- Metrics --------------------------------------------------------
    top1 = float(np.mean(pred_labels == true_labels))
    top5 = float(np.mean(np.any(top_idx == true_labels[:, None], axis=1)))

    per_class_correct = np.zeros(100)
    per_class_total = np.zeros(100)
    for t, p in zip(true_labels, pred_labels):
        per_class_total[int(t)] += 1
        if int(t) == int(p):
            per_class_correct[int(t)] += 1
    per_class_acc = per_class_correct / np.maximum(per_class_total, 1)

    confusion = np.zeros((100, 100), dtype=np.int32)
    for t, p in zip(true_labels, pred_labels):
        confusion[int(t), int(p)] += 1

    # ---- Save -----------------------------------------------------------
    npz_path = os.path.join(args.output_dir, "test_predictions.npz")
    np.savez(
        npz_path,
        true_labels=true_labels.astype(np.int32),
        pred_labels=pred_labels.astype(np.int32),
        top5_indices=top_idx.astype(np.int32),
        top5_probs=top_probs.astype(np.float32),
        per_class_acc=per_class_acc.astype(np.float32),
        confusion=confusion,
    )
    json_path = os.path.join(args.output_dir, "test_summary.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "top1_accuracy": top1,
                "top5_accuracy": top5,
                "num_test": int(top_idx.shape[0]),
                "checkpoint": args.checkpoint,
            },
            f,
            indent=2,
        )

    print(f"\nTop-1 accuracy: {top1 * 100:.2f}%")
    print(f"Top-5 accuracy: {top5 * 100:.2f}%")
    print(f"Saved {npz_path}")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
