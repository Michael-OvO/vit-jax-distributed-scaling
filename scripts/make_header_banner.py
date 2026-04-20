"""Render a clean, symbolic README header banner.

Left:  a schematic of the 8-chip TPU mesh synchronising into one ViT model.
Right: six real CIFAR-100 test images with their true class labels, loaded
       from the committed ``outputs/cifar100_test_raw.npz``.

No metrics, no table text --- purely a visual of *what this project is*.
Output: ``outputs/header.png``.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CIFAR100_CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy",
    "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail",
    "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper",
    "table", "tank", "telephone", "television", "tiger", "tractor", "train",
    "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf",
    "woman", "worm",
]
assert len(CIFAR100_CLASSES) == 100

# One visually distinctive example per broad category.
SHOWCASE_CLASSES = ["tiger", "sunflower", "bus", "skyscraper", "mushroom", "road"]


def _pick_indices(labels: np.ndarray, class_names: list[str], seed: int = 42) -> list[int]:
    rng = np.random.RandomState(seed)
    picks = []
    for name in class_names:
        cls = CIFAR100_CLASSES.index(name)
        matches = np.where(labels == cls)[0]
        picks.append(int(rng.choice(matches)))
    return picks


def _device_box(ax, x: float, y: float, label: str,
                fill: str = "#dbeafe", edge: str = "#2563eb") -> None:
    w, h = 0.95, 0.75
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            edgecolor=edge, facecolor=fill, linewidth=1.4,
        )
    )
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=10, color=edge, fontweight="bold",
            family="DejaVu Sans")


def _model_box(ax, x: float, y: float, w: float, h: float, label: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            edgecolor="#0f172a", facecolor="#0f172a", linewidth=1.5,
        )
    )
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=12, color="white", fontweight="bold",
            family="DejaVu Sans")


def _draw_schematic(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")

    # 8 TPU chips arranged in a 2x4 grid, reflecting the actual v6e-8 topology.
    chip_x0 = 1.3
    chip_y_top = 5.2
    chip_y_bot = 4.1
    col_w = 1.15
    chip_positions = []
    for col in range(4):
        x = chip_x0 + col * col_w
        _device_box(ax, x, chip_y_top, f"T{col}")
        _device_box(ax, x, chip_y_bot, f"T{col + 4}")
        chip_positions.append((x + 0.475, chip_y_bot))  # bottom centre of each column

    # A single model box below, representing the replicated ViT after all-reduce.
    model_x, model_y = 2.7, 1.3
    model_w, model_h = 4.6, 1.4
    _model_box(ax, model_x, model_y, model_w, model_h, "ViT-Small  (replicated)")

    # Converging arrows from every chip column down to the model's top edge.
    target_x_left = model_x + model_w * 0.35
    target_x_right = model_x + model_w * 0.65
    for i, (cx, cy) in enumerate(chip_positions):
        # Split the arrows so they visually funnel into the model.
        tx = target_x_left if i < 2 else target_x_right
        ax.annotate(
            "",
            xy=(tx, model_y + model_h),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle="-",
                color="#94a3b8",
                lw=1.0,
                alpha=0.55,
            ),
        )
    # Single bold sync arrow marking the all-reduce point.
    ax.annotate(
        "",
        xy=(model_x + model_w / 2, model_y + model_h + 0.02),
        xytext=(model_x + model_w / 2, chip_y_bot - 0.05),
        arrowprops=dict(arrowstyle="-|>", color="#2563eb", lw=2.2,
                        mutation_scale=16),
    )
    ax.text(
        model_x + model_w / 2 + 0.15, (chip_y_bot + model_y + model_h) / 2 - 0.05,
        "pmean\nall-reduce",
        ha="left", va="center", fontsize=9, color="#2563eb",
        fontweight="bold",
    )

    # Discreet caption.
    ax.text(
        5.0, 6.45, "8 TPU v6e chips  ·  synchronous data parallelism",
        ha="center", va="bottom", fontsize=11, color="#475569",
        style="italic",
    )


def _draw_thumbnails(fig, spec, images: np.ndarray, labels: np.ndarray,
                     indices: list[int]) -> None:
    inner = GridSpecFromSubplotSpec(
        2, len(indices), subplot_spec=spec, wspace=0.18, hspace=0.05,
        height_ratios=[0.9, 0.08],
    )
    for i, idx in enumerate(indices):
        ax = fig.add_subplot(inner[0, i])
        ax.imshow(images[idx], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cbd5e1")
            spine.set_linewidth(1.2)

        label_ax = fig.add_subplot(inner[1, i])
        label_ax.axis("off")
        name = CIFAR100_CLASSES[int(labels[idx])].replace("_", " ")
        label_ax.text(0.5, 0.6, name,
                      ha="center", va="top", fontsize=11,
                      color="#334155", family="DejaVu Sans")


def main() -> None:
    npz_path = os.path.join(ROOT, "outputs", "cifar100_test_raw.npz")
    data = np.load(npz_path)
    images = data["images"]
    labels = data["labels"]

    indices = _pick_indices(labels, SHOWCASE_CLASSES, seed=3)

    fig = plt.figure(figsize=(15, 3.6), facecolor="white")
    outer = GridSpec(
        1, 2, width_ratios=[2.1, 3.0], figure=fig,
        left=0.015, right=0.985, top=0.96, bottom=0.08, wspace=0.07,
    )

    ax_schema = fig.add_subplot(outer[0, 0])
    _draw_schematic(ax_schema)
    _draw_thumbnails(fig, outer[0, 1], images, labels, indices)

    output_path = os.path.join(ROOT, "outputs", "header.png")
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
