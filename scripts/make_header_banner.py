"""Render a wide README header banner that summarises the project visually.

Combines the title + author + headline metrics (left) with miniature versions
of the scaling-efficiency and straggler-slowdown plots (right), reading the
underlying numbers directly from the CSVs so the banner stays in sync with
whatever latest run is committed.

Output: ``outputs/header.png``.
"""

from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read_csv(rel_path: str) -> list[dict]:
    with open(os.path.join(ROOT, rel_path)) as f:
        return list(csv.DictReader(f))


def main() -> None:
    scaling = _read_csv("outputs/scaling/scaling_results.csv")
    straggler = _read_csv("outputs/straggler/straggler_results.csv")

    devices = [int(r["num_devices"]) for r in scaling]
    efficiency = [float(r["scaling_efficiency"]) for r in scaling]

    # Straggler CSV has a baseline row where ``delay_iterations`` is empty.
    delays = [int(r["delay_iterations"]) for r in straggler if r["delay_iterations"]]
    slowdowns = [float(r["slowdown_factor"]) for r in straggler if r["delay_iterations"]]

    fig = plt.figure(figsize=(16, 4.5), facecolor="white")
    gs = GridSpec(
        1, 3, width_ratios=[2.1, 1.0, 1.0], figure=fig,
        left=0.02, right=0.98, top=0.92, bottom=0.14, wspace=0.28,
    )

    # ---- Title column ----
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis("off")

    ax_title.text(
        0.02, 0.88,
        "ViT-JAX Distributed Scaling",
        fontsize=30, fontweight="bold", color="#0f172a",
        va="top", ha="left",
    )
    ax_title.text(
        0.02, 0.63,
        "Data-parallel Vision Transformer on TPU v6e-8",
        fontsize=15, color="#475569", va="top", ha="left", style="italic",
    )

    # Headline metric strip
    metrics = [
        ("55.04%", "top-1 test acc"),
        ("80.18%", "top-5 test acc"),
        ("3.33x",  "straggler slowdown"),
        ("15K/s",  "throughput (imgs)"),
    ]
    for i, (value, label) in enumerate(metrics):
        x = 0.025 + i * 0.235
        ax_title.text(
            x, 0.38, value,
            fontsize=22, fontweight="bold", color="#2563eb",
            va="top", ha="left", family="DejaVu Sans",
        )
        ax_title.text(
            x, 0.22, label,
            fontsize=11, color="#64748b", va="top", ha="left",
        )

    ax_title.text(
        0.02, 0.02,
        "CS 390: Distributed Systems  ·  Duke University Spring 2026  ·  Jinao Wang",
        fontsize=11, color="#94a3b8", va="bottom", ha="left",
    )

    # ---- Scaling efficiency mini-plot ----
    ax_scale = fig.add_subplot(gs[0, 1])
    ax_scale.plot(
        devices, [e * 100 for e in efficiency],
        "o-", linewidth=2.8, markersize=9, color="#2563eb",
    )
    ax_scale.axhline(100, color="#94a3b8", linestyle="--", linewidth=1)
    ax_scale.set_xticks(devices)
    ax_scale.set_ylim(0, 115)
    ax_scale.set_xlabel("Devices", fontsize=11)
    ax_scale.set_ylabel("Efficiency (%)", fontsize=11)
    ax_scale.set_title("Scaling efficiency", fontsize=13, fontweight="bold", pad=8)
    ax_scale.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax_scale.spines[spine].set_visible(False)
    ax_scale.tick_params(labelsize=10)

    # ---- Straggler slowdown mini-plot ----
    ax_strag = fig.add_subplot(gs[0, 2])
    ax_strag.plot(
        delays, slowdowns,
        "o-", linewidth=2.8, markersize=9, color="#dc2626",
    )
    ax_strag.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=1)
    ax_strag.set_xscale("log")
    ax_strag.set_xlabel("Delay iters (log)", fontsize=11)
    ax_strag.set_ylabel("Slowdown (×)", fontsize=11)
    ax_strag.set_title("Slowdown from 1 slow device", fontsize=13, fontweight="bold", pad=8)
    ax_strag.grid(True, alpha=0.25)
    for spine in ("top", "right"):
        ax_strag.spines[spine].set_visible(False)
    ax_strag.tick_params(labelsize=10)

    output_path = os.path.join(ROOT, "outputs", "header.png")
    fig.savefig(output_path, dpi=120, facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
