"""Scaling experiment: measure how ViT training throughput scales with device count.

Runs a short benchmark of training steps, records step time and throughput,
and saves results + plots for comparison across different device configurations.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax

from vit_jax_distributed.models.vit import ViTSmall
from vit_jax_distributed.data.cifar100 import get_datasets, shard_batch
from vit_jax_distributed.distributed.parallel import (
    create_train_step,
    replicate_state,
    create_train_state,
)
from vit_jax_distributed.utils.timing import StepTimer
from vit_jax_distributed.experiments.common import ExperimentConfig, SimpleDataConfig


@dataclass
class ScalingConfig(ExperimentConfig):
    """Configuration for the scaling experiment."""

    output_dir: str = "results/scaling"


def run_scaling_experiment(config: ScalingConfig) -> dict:
    """Run the scaling benchmark and save results.

    Benchmarks training throughput using the available (or configured) number
    of devices. Results are saved to CSV, JSON, and PNG plots in
    ``config.output_dir``.

    Parameters
    ----------
    config : ScalingConfig
        Experiment configuration.

    Returns
    -------
    dict
        A results dictionary with per-device-count measurements.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    all_devices = jax.local_devices()
    max_devices = len(all_devices)

    # Determine which device counts to benchmark.
    if config.num_devices is not None:
        device_counts = [min(config.num_devices, max_devices)]
    else:
        # Powers of 2 up to max_devices.
        device_counts = []
        n = 1
        while n <= max_devices:
            device_counts.append(n)
            n *= 2
        # Always include max_devices if it's not already there.
        if max_devices not in device_counts:
            device_counts.append(max_devices)
        device_counts.sort()

    print(f"Available devices: {max_devices} ({all_devices[0].platform})")
    print(f"Device counts to benchmark: {device_counts}")
    print(f"Benchmark steps: {config.benchmark_steps} (warmup: {config.warmup_benchmark_steps})")
    print()

    results: List[dict] = []

    for num_devices in device_counts:
        print(f"--- Benchmarking with {num_devices} device(s) ---")

        # Effective batch size must be divisible by num_devices.
        effective_batch_size = (config.batch_size // num_devices) * num_devices
        print(f"  Effective batch size: {effective_batch_size}")

        # Subset of devices we actually map across for this run. Must be
        # threaded into both pmap and the replication helper; otherwise they
        # default to *all* local devices and the batch leading-dim fails to
        # match.
        selected_devices = all_devices[:num_devices]

        # Create model and training state.
        rng = jax.random.PRNGKey(config.seed)
        rng, init_rng = jax.random.split(rng)
        model = ViTSmall(image_size=config.image_size)
        state = create_train_state(
            init_rng,
            model,
            config.learning_rate,
            image_size=config.image_size,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps,
        )

        state = replicate_state(state, devices=selected_devices)

        # Create data iterator.
        # Build a minimal config object for get_datasets.
        data_cfg = SimpleDataConfig(
            batch_size=effective_batch_size,
            image_size=config.image_size,
            data_augmentation=config.data_augmentation,
        )
        train_iter, _ = get_datasets(data_cfg, num_devices=num_devices, seed=config.seed)

        # Create pmap'd train step bound to this device subset.
        train_step_fn = create_train_step(devices=selected_devices)

        step_timer = StepTimer()

        # Warmup steps (not timed).
        print(f"  Running {config.warmup_benchmark_steps} warmup steps...")
        for _ in range(config.warmup_benchmark_steps):
            batch = next(train_iter)
            batch = shard_batch(batch, num_devices)
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_devices)
            state, _ = train_step_fn(state, batch, step_rngs)

        # Block until warmup computation is done.
        jax.tree.map(lambda x: x.block_until_ready(), state)

        # Benchmark steps.
        print(f"  Running {config.benchmark_steps} benchmark steps...")
        step_timer.reset()
        for i in range(config.benchmark_steps):
            step_timer.tick()
            batch = next(train_iter)
            batch = shard_batch(batch, num_devices)
            rng, step_rng = jax.random.split(rng)
            step_rngs = jax.random.split(step_rng, num_devices)
            state, metrics = train_step_fn(state, batch, step_rngs)
            # Block to get accurate timing.
            jax.tree.map(lambda x: x.block_until_ready(), state)

        # Final tick to capture the last interval.
        step_timer.tick()

        stats = step_timer.get_stats()
        throughput = step_timer.get_throughput(effective_batch_size)

        # Compute scaling efficiency relative to single-device baseline.
        # Will be filled in after all runs.
        result = {
            "num_devices": num_devices,
            "batch_size": effective_batch_size,
            "mean_step_time": stats.get("mean", 0.0),
            "std_step_time": stats.get("std", 0.0),
            "min_step_time": stats.get("min", 0.0),
            "max_step_time": stats.get("max", 0.0),
            "p50_step_time": stats.get("p50", 0.0),
            "p95_step_time": stats.get("p95", 0.0),
            "p99_step_time": stats.get("p99", 0.0),
            "throughput_img_per_sec": throughput,
            "final_loss": float(metrics["loss"][0]),
            "final_accuracy": float(metrics["accuracy"][0]),
        }
        results.append(result)

        print(f"  Mean step time: {result['mean_step_time']:.4f}s")
        print(f"  Throughput: {throughput:.1f} images/sec")
        print()

    # Compute scaling efficiency (relative to single-device throughput).
    if results:
        base_throughput = results[0]["throughput_img_per_sec"]
        base_devices = results[0]["num_devices"]
        for r in results:
            if base_throughput > 0:
                ideal_throughput = base_throughput * (r["num_devices"] / base_devices)
                r["scaling_efficiency"] = r["throughput_img_per_sec"] / ideal_throughput
            else:
                r["scaling_efficiency"] = 0.0

    # Save results.
    _save_results(results, config.output_dir)
    _generate_plots(results, config.output_dir)
    _print_summary_table(results)

    return {"device_counts": device_counts, "results": results}


def generate_scaling_plots(results_dir: str) -> None:
    """Re-generate plots from previously saved JSON results.

    Parameters
    ----------
    results_dir : str
        Directory containing ``scaling_results.json``.
    """
    json_path = os.path.join(results_dir, "scaling_results.json")
    if not os.path.exists(json_path):
        print(f"Results file not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    results = data["results"]
    _generate_plots(results, results_dir)
    print(f"Plots regenerated in {results_dir}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_results(results: List[dict], output_dir: str) -> None:
    """Save results to CSV and JSON."""
    import csv

    # CSV
    csv_path = os.path.join(output_dir, "scaling_results.csv")
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_path}")

    # JSON
    json_path = os.path.join(output_dir, "scaling_results.json")
    data = {
        "experiment": "scaling",
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {json_path}")


def _generate_plots(results: List[dict], output_dir: str) -> None:
    """Generate throughput, step time, and scaling efficiency plots."""
    if not results:
        return

    device_counts = [r["num_devices"] for r in results]
    throughputs = [r["throughput_img_per_sec"] for r in results]
    step_times = [r["mean_step_time"] for r in results]
    efficiencies = [r.get("scaling_efficiency", 0.0) for r in results]

    # -- Throughput vs num_devices --
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(device_counts, throughputs, "o-", linewidth=2, markersize=8, label="Measured")
    # Ideal linear scaling line.
    if throughputs:
        base = throughputs[0]
        base_n = device_counts[0]
        ideal = [base * (n / base_n) for n in device_counts]
        ax.plot(device_counts, ideal, "--", color="gray", linewidth=1.5, label="Ideal linear")
    ax.set_xlabel("Number of Devices")
    ax.set_ylabel("Throughput (images/sec)")
    ax.set_title("Training Throughput vs. Device Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if len(device_counts) > 1:
        ax.set_xticks(device_counts)
    fig.tight_layout()
    path = os.path.join(output_dir, "throughput_vs_devices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")

    # -- Step time vs num_devices --
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        [str(n) for n in device_counts],
        step_times,
        color="steelblue",
        edgecolor="black",
    )
    ax.set_xlabel("Number of Devices")
    ax.set_ylabel("Mean Step Time (s)")
    ax.set_title("Training Step Time vs. Device Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "step_time_vs_devices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")

    # -- Scaling efficiency vs num_devices --
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(device_counts, efficiencies, "s-", linewidth=2, markersize=8, color="green", label="Measured")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, label="Ideal (100%)")
    ax.set_xlabel("Number of Devices")
    ax.set_ylabel("Scaling Efficiency")
    ax.set_title("Scaling Efficiency vs. Device Count")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if len(device_counts) > 1:
        ax.set_xticks(device_counts)
    fig.tight_layout()
    path = os.path.join(output_dir, "scaling_efficiency_vs_devices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def _print_summary_table(results: List[dict]) -> None:
    """Print a formatted summary table to stdout."""
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 85)
    print("SCALING EXPERIMENT RESULTS")
    print("=" * 85)

    header = (
        f"{'Devices':>8}  {'Batch':>8}  {'Step Time (s)':>14}  "
        f"{'Throughput':>12}  {'Efficiency':>11}"
    )
    print(header)
    print("-" * 85)

    for r in results:
        print(
            f"{r['num_devices']:>8}  {r['batch_size']:>8}  "
            f"{r['mean_step_time']:>14.4f}  "
            f"{r['throughput_img_per_sec']:>10.1f}/s  "
            f"{r.get('scaling_efficiency', 0.0):>10.1%}"
        )

    print("=" * 85)
    print()
