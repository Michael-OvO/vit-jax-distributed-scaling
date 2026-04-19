"""Core training loop for distributed ViT training on CIFAR-100.

Orchestrates model creation, data loading, distributed training with pmap,
evaluation, logging, and metric export.
"""

import os
import time
import logging

import jax
import jax.numpy as jnp
import numpy as np

from vit_jax_distributed.models.vit import ViTSmall
from vit_jax_distributed.data.cifar100 import get_datasets, shard_batch, get_num_examples
from vit_jax_distributed.distributed.parallel import (
    create_train_step,
    create_eval_step,
    replicate_state,
    unreplicate_state,
    create_train_state,
)
from vit_jax_distributed.utils.logging import MetricsLogger
from vit_jax_distributed.utils.timing import Timer, StepTimer
from vit_jax_distributed.utils.config import config_to_dict
from vit_jax_distributed.utils.checkpoint import save_replicated

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _detect_devices(config):
    """Return the list of JAX devices to use based on config.num_devices.

    If ``config.num_devices`` is 0 or unset, all local devices are used.
    If more devices are requested than available, a warning is logged and
    all available devices are used.
    """
    all_devices = jax.local_devices()
    requested = config.num_devices

    if requested <= 0:
        return all_devices

    if requested > len(all_devices):
        logger.warning(
            "Requested %d devices but only %d available. Using all %d.",
            requested,
            len(all_devices),
            len(all_devices),
        )
        return all_devices

    return all_devices[:requested]


def _create_model(config):
    """Instantiate a ViTSmall module from config attributes."""
    return ViTSmall(
        num_classes=100,
        patch_size=config.patch_size,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        mlp_dim=config.mlp_dim,
        dropout_rate=config.dropout_rate,
        image_size=config.image_size,
    )


def _make_test_iter(config, num_devices):
    """Create a fresh single-pass test iterator (reuses cached TFDS data)."""
    _, test_iter = get_datasets(config, num_devices=num_devices, seed=config.seed + 1)
    return test_iter


def _run_evaluation(state, evaluation_step, config, num_devices):
    """Run one full pass over the test set and return averaged metrics."""
    num_test = get_num_examples("test")
    batch_size = (config.batch_size // num_devices) * num_devices
    num_evaluation_steps = num_test // batch_size

    test_iter = _make_test_iter(config, num_devices)

    total_loss = 0.0
    total_acc = 0.0
    count = 0

    for _ in range(num_evaluation_steps):
        try:
            batch = next(test_iter)
        except StopIteration:
            break
        batch = shard_batch(batch, num_devices)
        metrics = evaluation_step(state, batch)
        total_loss += float(metrics["loss"][0])
        total_acc += float(metrics["accuracy"][0])
        count += 1

    if count == 0:
        return {"test_loss": 0.0, "test_accuracy": 0.0}

    return {
        "test_loss": total_loss / count,
        "test_accuracy": total_acc / count,
    }


def train(config):
    """Main training entry point.

    Args:
        config: An ``argparse.Namespace`` (or similar object) with training
            hyper-parameters.  See :func:`vit_jax_distributed.utils.config.get_config`
            for the full list of supported attributes.

    Returns:
        A dict with final metrics, step-time statistics, and throughput.
    """
    # ------------------------------------------------------------------
    # 1. Setup: devices, output directory, logging
    # ------------------------------------------------------------------
    devices = _detect_devices(config)
    num_devices = len(devices)

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    config_dict = config_to_dict(config)
    print(f"\n{'=' * 60}")
    print("JAX Distributed ViT Training")
    print(f"{'=' * 60}")
    print(f"Devices:      {num_devices} x {devices[0].platform}")
    print(f"Backend:      {jax.default_backend()}")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # 2. Model and training schedule
    # ------------------------------------------------------------------
    model = _create_model(config)

    num_train = get_num_examples("train")
    batch_size = (config.batch_size // num_devices) * num_devices
    steps_per_epoch = num_train // batch_size
    total_steps = steps_per_epoch * config.epochs

    print(f"Training examples: {num_train}")
    print(f"Effective batch size: {batch_size}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")

    # ------------------------------------------------------------------
    # 3. Create train state and replicate across devices
    # ------------------------------------------------------------------
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(
        init_rng,
        model,
        learning_rate=config.learning_rate,
        image_size=config.image_size,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
    )

    param_count = sum(p.size for p in jax.tree.leaves(state.params))
    print(f"Model parameters: {param_count:,}\n")

    # Pass the selected subset explicitly to every pmap/replicate call, so
    # --num_devices < all-local works on multi-device hosts (e.g. TPU v3-8).
    state = replicate_state(state, devices=devices)

    # ------------------------------------------------------------------
    # 4. Data iterators
    # ------------------------------------------------------------------
    train_iter, _ = get_datasets(config, num_devices=num_devices, seed=config.seed)

    # ------------------------------------------------------------------
    # 5. Distributed step functions
    # ------------------------------------------------------------------
    p_train_step = create_train_step(devices=devices)
    p_evaluation_step = create_eval_step(devices=devices)

    # ------------------------------------------------------------------
    # 6. Metrics and timing
    # ------------------------------------------------------------------
    metrics_logger = MetricsLogger(
        config=config_dict,
        num_devices=num_devices,
    )
    step_timer = StepTimer()

    log_every = config.log_every
    run_test_every = config.eval_every

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    global_step = 0

    with Timer() as wall_timer:
        for epoch in range(config.epochs):
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_count = 0

            if _HAS_TQDM:
                pbar = tqdm(
                    range(steps_per_epoch),
                    desc=f"Epoch {epoch + 1}/{config.epochs}",
                    leave=True,
                )
            else:
                pbar = range(steps_per_epoch)

            for step_in_epoch in pbar:
                # --- Get batch and shard ---
                batch = next(train_iter)
                batch = shard_batch(batch, num_devices)

                # --- RNG for this step (one sub-key per device) ---
                rng, step_rng = jax.random.split(rng)
                step_rngs = jax.random.split(step_rng, num_devices)

                # --- Train step ---
                state, metrics = p_train_step(state, batch, step_rngs)

                # Extract scalar metrics (pmean'd, so device 0 is representative)
                step_loss = float(metrics["loss"][0])
                step_acc = float(metrics["accuracy"][0])

                epoch_loss += step_loss
                epoch_acc += step_acc
                epoch_count += 1

                # --- Logging ---
                if global_step % log_every == 0:
                    step_stats = step_timer.get_stats()
                    step_time_ms = step_stats.get("mean", 0.0) * 1000

                    log_dict = {
                        "train_loss": step_loss,
                        "train_accuracy": step_acc,
                        "epoch": epoch + 1,
                        "step_time_ms": step_time_ms,
                    }
                    metrics_logger.log_step(global_step, log_dict)

                    if _HAS_TQDM:
                        pbar.set_postfix(
                            loss=f"{step_loss:.4f}",
                            acc=f"{step_acc:.4f}",
                            step_ms=f"{step_time_ms:.1f}",
                        )
                    else:
                        print(
                            f"  [Step {global_step:>6d}] "
                            f"loss={step_loss:.4f}  "
                            f"acc={step_acc:.4f}  "
                            f"step_time={step_time_ms:.1f}ms"
                        )

                step_timer.tick()
                global_step += 1

            # --- End of epoch ---
            avg_epoch_loss = epoch_loss / max(epoch_count, 1)
            avg_epoch_acc = epoch_acc / max(epoch_count, 1)

            epoch_metrics = {
                "train_loss": avg_epoch_loss,
                "train_accuracy": avg_epoch_acc,
            }

            # --- Test-set assessment ---
            if (epoch + 1) % run_test_every == 0:
                print(f"\nRunning test-set assessment after epoch {epoch + 1}...")
                test_metrics = _run_evaluation(
                    state, p_evaluation_step, config, num_devices
                )
                epoch_metrics.update(test_metrics)
                print(
                    f"  Test loss: {test_metrics['test_loss']:.4f}  "
                    f"Test accuracy: {test_metrics['test_accuracy']:.4f}"
                )

            metrics_logger.log_epoch(epoch + 1, epoch_metrics)
            print(
                f"Epoch {epoch + 1}/{config.epochs} -- "
                f"train_loss={avg_epoch_loss:.4f}  "
                f"train_acc={avg_epoch_acc:.4f}"
            )

            # --- Checkpoint after each epoch ---
            checkpoint_path = save_replicated(
                state,
                output_dir,
                step=global_step,
                metadata={
                    "epoch": epoch + 1,
                    "total_epochs": config.epochs,
                    "config": config_dict,
                    "epoch_metrics": epoch_metrics,
                },
            )
            print(f"  Checkpoint saved: {checkpoint_path}")

    total_wall_time = wall_timer.elapsed

    # ------------------------------------------------------------------
    # 8. Save metrics and print summary
    # ------------------------------------------------------------------
    step_csv_path = os.path.join(output_dir, "step_metrics.csv")
    epoch_csv_path = os.path.join(output_dir, "epoch_metrics.csv")
    json_path = os.path.join(output_dir, "training_log.json")

    metrics_logger.save_csv(step_csv_path, which="step")
    metrics_logger.save_csv(epoch_csv_path, which="epoch")
    metrics_logger.save_json(json_path)

    metrics_logger.print_summary()

    print(f"Metrics saved to {output_dir}")
    print(f"  - {step_csv_path}")
    print(f"  - {epoch_csv_path}")
    print(f"  - {json_path}")

    # ------------------------------------------------------------------
    # 9. Build and return results dict
    # ------------------------------------------------------------------
    step_stats = step_timer.get_stats()
    throughput = step_timer.get_throughput(batch_size)

    final_epoch = metrics_logger.epoch_logs[-1] if metrics_logger.epoch_logs else {}

    results = {
        "final_train_loss": final_epoch.get("train_loss", None),
        "final_train_accuracy": final_epoch.get("train_accuracy", None),
        "final_test_loss": final_epoch.get("test_loss", None),
        "final_test_accuracy": final_epoch.get("test_accuracy", None),
        "total_steps": global_step,
        "total_epochs": config.epochs,
        "wall_time_seconds": total_wall_time,
        "step_time_stats": step_stats,
        "throughput_images_per_sec": throughput,
        "num_devices": num_devices,
        "num_parameters": param_count,
        "output_dir": output_dir,
    }

    print(f"\nTotal wall time: {total_wall_time:.1f}s")
    print(f"Throughput: {throughput:.1f} images/sec")
    if step_stats:
        print(
            f"Step time: {step_stats['mean'] * 1000:.1f}ms "
            f"(p50={step_stats['p50'] * 1000:.1f}ms, "
            f"p95={step_stats['p95'] * 1000:.1f}ms)"
        )

    return results
