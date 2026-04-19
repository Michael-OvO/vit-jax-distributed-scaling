"""Straggler simulation experiment: measure impact of a slow device on training.

Benchmarks normal training vs. training with an artificial straggler on
device 0. Tests multiple delay magnitudes and records the slowdown factor.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import numpy as np

from vit_jax_distributed.models.vit import ViTSmall
from vit_jax_distributed.data.cifar100 import get_datasets, shard_batch
from vit_jax_distributed.distributed.parallel import (
    create_train_step,
    create_train_step_with_straggler,
    replicate_state,
    create_train_state,
)
from vit_jax_distributed.utils.timing import StepTimer
from vit_jax_distributed.experiments.common import ExperimentConfig, SimpleDataConfig


@dataclass
class StragglerConfig(ExperimentConfig):
    """Configuration for the straggler experiment."""

    straggler_delay: int = 1000
    delay_values: Optional[List[int]] = None
    output_dir: str = "results/straggler"


def run_straggler_experiment(config: StragglerConfig) -> dict:
    """Run the straggler benchmark and save results.

    Compares baseline training step time against training with an artificial
    straggler (extra compute on device 0) at various delay magnitudes.

    Parameters
    ----------
    config : StragglerConfig
        Experiment configuration.

    Returns
    -------
    dict
        Results dictionary with baseline and per-delay measurements.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    all_devices = jax.local_devices()
    num_devices = config.num_devices or len(all_devices)
    num_devices = min(num_devices, len(all_devices))
    selected_devices = all_devices[:num_devices]

    if num_devices < 2:
        print("WARNING: Straggler experiment requires >= 2 devices.")
        print(f"Only {num_devices} device(s) available. Running baseline only.")

    print(f"Devices: {num_devices} ({all_devices[0].platform})")
    print(f"Benchmark steps: {config.benchmark_steps} (warmup: {config.warmup_benchmark_steps})")
    print()

    # Effective batch size.
    effective_batch_size = (config.batch_size // num_devices) * num_devices

    # Delay values to test. Calibrated for TPU v6e (~918 TFLOPs bf16) using
    # 512x512 matmul body in create_train_step_with_straggler — the old
    # values [100 ... 10000] produce microsecond-scale delays on modern
    # hardware and show up as noise. Scaled range below spans from a ~1 ms
    # injection (visible in p50) up to ~60 ms (~2x slowdown).
    delay_values = config.delay_values or [5000, 20000, 50000, 100000, 200000]

    # --- Setup shared resources ---
    rng = jax.random.PRNGKey(config.seed)
    model = ViTSmall(image_size=config.image_size)

    data_cfg = SimpleDataConfig(
        batch_size=effective_batch_size,
        image_size=config.image_size,
        data_augmentation=config.data_augmentation,
    )

    # --- Baseline benchmark ---
    print("=== Baseline (no straggler) ===")
    baseline_stats = _run_benchmark(
        config=config,
        rng=rng,
        model=model,
        data_cfg=data_cfg,
        num_devices=num_devices,
        devices=selected_devices,
        train_step_fn=create_train_step(devices=selected_devices),
        label="baseline",
    )
    print(f"  Mean step time: {baseline_stats['mean_step_time']:.4f}s")
    print(f"  Throughput: {baseline_stats['throughput_img_per_sec']:.1f} images/sec")
    print()

    # --- Straggler benchmarks ---
    straggler_results: List[dict] = []

    if num_devices >= 2:
        for delay in delay_values:
            print(f"=== Straggler (delay_iterations={delay}) ===")
            straggler_step_fn = create_train_step_with_straggler(
                delay_iterations=delay,
                devices=selected_devices,
            )
            stats = _run_benchmark(
                config=config,
                rng=rng,
                model=model,
                data_cfg=data_cfg,
                num_devices=num_devices,
                devices=selected_devices,
                train_step_fn=straggler_step_fn,
                label=f"straggler_{delay}",
            )

            slowdown = stats["mean_step_time"] / baseline_stats["mean_step_time"] if baseline_stats["mean_step_time"] > 0 else 0.0
            stats["delay_iterations"] = delay
            stats["slowdown_factor"] = slowdown

            straggler_results.append(stats)

            print(f"  Mean step time: {stats['mean_step_time']:.4f}s")
            print(f"  Slowdown factor: {slowdown:.2f}x")
            print()
    else:
        print("Skipping straggler benchmarks (only 1 device available).")
        print()

    # Assemble full results.
    results = {
        "experiment": "straggler",
        "num_devices": num_devices,
        "batch_size": effective_batch_size,
        "baseline": baseline_stats,
        "straggler_runs": straggler_results,
    }

    # Save and plot.
    _save_results(results, config.output_dir)
    _generate_plots(baseline_stats, straggler_results, config.output_dir)
    _print_analysis_table(baseline_stats, straggler_results, num_devices)

    return results


def _run_benchmark(
    config: StragglerConfig,
    rng: jax.Array,
    model,
    data_cfg,
    num_devices: int,
    devices,
    train_step_fn,
    label: str,
) -> dict:
    """Run a short training benchmark and return timing statistics.

    Parameters
    ----------
    config : StragglerConfig
        Experiment configuration.
    rng : jax.Array
        PRNG key (will be split internally).
    model : ViTSmall
        The model to train.
    data_cfg : _SimpleConfig
        Data configuration for get_datasets.
    num_devices : int
        Number of devices.
    train_step_fn : callable
        The pmap'd training step function.
    label : str
        Human-readable label for this benchmark run.

    Returns
    -------
    dict
        Timing statistics and throughput.
    """
    effective_batch_size = data_cfg.batch_size

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        init_rng,
        model,
        config.learning_rate,
        image_size=config.image_size,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
    )
    state = replicate_state(state, devices=devices)

    train_iter, _ = get_datasets(data_cfg, num_devices=num_devices, seed=config.seed)
    step_timer = StepTimer()

    # Warmup (not timed).
    for _ in range(config.warmup_benchmark_steps):
        batch = next(train_iter)
        batch = shard_batch(batch, num_devices)
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_devices)
        state, _ = train_step_fn(state, batch, step_rngs)

    # Block until warmup is done.
    jax.tree.map(lambda x: x.block_until_ready(), state)

    # Timed benchmark.
    step_timer.reset()
    for _ in range(config.benchmark_steps):
        step_timer.tick()
        batch = next(train_iter)
        batch = shard_batch(batch, num_devices)
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_devices)
        state, metrics = train_step_fn(state, batch, step_rngs)
        jax.tree.map(lambda x: x.block_until_ready(), state)

    # Final tick.
    step_timer.tick()

    stats = step_timer.get_stats()
    throughput = step_timer.get_throughput(effective_batch_size)

    return {
        "label": label,
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


# ---------------------------------------------------------------------------
# Save / plot / print helpers
# ---------------------------------------------------------------------------

def _save_results(results: dict, output_dir: str) -> None:
    """Save results to JSON and CSV."""
    import csv

    # JSON
    json_path = os.path.join(output_dir, "straggler_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")

    # CSV with one row per benchmark run (baseline + each delay).
    csv_path = os.path.join(output_dir, "straggler_results.csv")
    rows = [results["baseline"]]
    for sr in results.get("straggler_runs", []):
        rows.append(sr)

    if rows:
        fieldnames = list(rows[-1].keys())
        # Ensure all keys present.
        for row in rows:
            for k in row:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {csv_path}")


def _generate_plots(
    baseline: dict,
    straggler_runs: List[dict],
    output_dir: str,
) -> None:
    """Generate comparison bar chart and slowdown line plot."""
    if not straggler_runs:
        # Only baseline available; nothing interesting to plot.
        return

    # -- Step time comparison bar chart --
    labels = ["Baseline"] + [f"Delay {r['delay_iterations']}" for r in straggler_runs]
    times = [baseline["mean_step_time"]] + [r["mean_step_time"] for r in straggler_runs]
    stds = [baseline["std_step_time"]] + [r["std_step_time"] for r in straggler_runs]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, times, yerr=stds, capsize=4, color="steelblue", edgecolor="black")
    # Highlight baseline bar.
    bars[0].set_color("seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Mean Step Time (s)")
    ax.set_title("Step Time: Baseline vs. Straggler Variants")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "step_time_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")

    # -- Slowdown factor vs delay_iterations --
    delays = [r["delay_iterations"] for r in straggler_runs]
    slowdowns = [r["slowdown_factor"] for r in straggler_runs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(delays, slowdowns, "o-", linewidth=2, markersize=8, color="crimson")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.5, label="No slowdown")
    ax.set_xlabel("Delay Iterations (straggler compute)")
    ax.set_ylabel("Slowdown Factor (x)")
    ax.set_title("Straggler Slowdown vs. Delay Magnitude")
    ax.set_xscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "slowdown_vs_delay.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Plot saved: {path}")


def _print_analysis_table(
    baseline: dict,
    straggler_runs: List[dict],
    num_devices: int,
) -> None:
    """Print a formatted analysis table to stdout."""
    print("\n" + "=" * 90)
    print("STRAGGLER EXPERIMENT RESULTS")
    print(f"Devices: {num_devices}")
    print("=" * 90)

    header = (
        f"{'Run':>20}  {'Step Time (s)':>14}  {'Std (s)':>10}  "
        f"{'Throughput':>12}  {'Slowdown':>10}"
    )
    print(header)
    print("-" * 90)

    # Baseline row.
    print(
        f"{'Baseline':>20}  {baseline['mean_step_time']:>14.4f}  "
        f"{baseline['std_step_time']:>10.4f}  "
        f"{baseline['throughput_img_per_sec']:>10.1f}/s  "
        f"{'1.00x':>10}"
    )

    # Straggler rows.
    for r in straggler_runs:
        label = f"Delay {r['delay_iterations']}"
        print(
            f"{label:>20}  {r['mean_step_time']:>14.4f}  "
            f"{r['std_step_time']:>10.4f}  "
            f"{r['throughput_img_per_sec']:>10.1f}/s  "
            f"{r['slowdown_factor']:>9.2f}x"
        )

    print("=" * 90)

    if straggler_runs:
        max_slowdown = max(r["slowdown_factor"] for r in straggler_runs)
        max_delay = max(r["delay_iterations"] for r in straggler_runs if r["slowdown_factor"] == max_slowdown)
        print(f"\nMax slowdown: {max_slowdown:.2f}x at delay_iterations={max_delay}")
        print(
            "A single straggler device forces all other devices to wait at "
            "the all-reduce barrier,\ndemonstrating the synchronous training "
            "bottleneck."
        )
    print()
