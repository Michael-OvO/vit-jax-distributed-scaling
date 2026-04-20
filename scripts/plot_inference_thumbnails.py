"""Side-by-side inference figure: CIFAR test thumbnails + model top-3.

Produces a grid of CIFAR-100 test images with each tile showing:
  - the image itself (un-standardised back to [0, 1])
  - the true class (header)
  - the model's top-3 predictions with probabilities
  - colour: green border if top-1 matches truth, red if not

Samples are mixed: half correct, half wrong, for maximum presentation value.

Input: ``test_predictions.npz`` from ``full_test_eval.py``.
"""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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


def _load_display_images(images_npz: str | None) -> np.ndarray:
    """Return (10000, 32, 32, 3) float32 in [0, 1] ready for matplotlib.

    Prefers a local uint8 NPZ if provided (no tfds dependency). Otherwise
    falls back to importing the training data pipeline (which requires
    ``tensorflow-datasets``).
    """
    if images_npz is not None:
        data = np.load(images_npz)
        imgs = data["images"]
        if imgs.dtype == np.uint8:
            return imgs.astype(np.float32) / 255.0
        return np.clip(imgs, 0.0, 1.0)

    # Fallback: let the training pipeline load + standardise via tfds,
    # then invert the standardisation for display.
    from vit_jax_distributed.data.cifar100 import (
        _load_split, CIFAR100_MEAN, CIFAR100_STD,
    )
    std_imgs, _ = _load_split("test")
    return np.clip(std_imgs * CIFAR100_STD + CIFAR100_MEAN, 0.0, 1.0)


def _short_label(name: str, max_len: int = 12) -> str:
    name = name.replace("_", " ")
    return name if len(name) <= max_len else name[:max_len - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_npz")
    parser.add_argument("--output", default=None,
                        help="Output PNG path; if omitted, writes "
                             "inference_thumbnails.png next to the NPZ.")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mix", action="store_true", default=True,
                        help="Half correct, half wrong (default: on)")
    parser.add_argument("--images_npz", default=None,
                        help="Path to uint8 NPZ of raw CIFAR-100 test images "
                             "(keys: 'images', 'labels'). If omitted, loads "
                             "via tensorflow-datasets.")
    args = parser.parse_args()

    data = np.load(args.predictions_npz)
    true = data["true_labels"]
    pred = data["pred_labels"]
    top5 = data["top5_indices"]
    probs = data["top5_probs"]

    print("Loading CIFAR-100 test images for display...")
    images = _load_display_images(args.images_npz)

    correct_mask = (pred == true)
    n_cells = args.rows * args.cols
    rng = np.random.RandomState(args.seed)

    if args.mix and (correct_mask.any() and (~correct_mask).any()):
        n_correct = n_cells // 2
        n_wrong = n_cells - n_correct
        c_idx = rng.choice(np.where(correct_mask)[0],
                           min(n_correct, int(correct_mask.sum())), replace=False)
        w_idx = rng.choice(np.where(~correct_mask)[0],
                           min(n_wrong, int((~correct_mask).sum())), replace=False)
        picked = np.concatenate([c_idx, w_idx])
        rng.shuffle(picked)
    else:
        picked = rng.choice(len(true), n_cells, replace=False)

    fig, axes = plt.subplots(args.rows, args.cols,
                             figsize=(2.2 * args.cols, 3.0 * args.rows))
    axes = np.atleast_2d(axes)

    for cell, idx in enumerate(picked):
        r, c = divmod(cell, args.cols)
        ax = axes[r, c]
        ax.imshow(images[idx], interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])

        ok = int(pred[idx]) == int(true[idx])
        border = "#2ca02c" if ok else "#d62728"
        for spine in ax.spines.values():
            spine.set_edgecolor(border)
            spine.set_linewidth(3)

        true_name = CIFAR100_CLASSES[int(true[idx])]
        ax.set_title(f"true: {_short_label(true_name)}", fontsize=9, pad=4,
                     color="#333333")

        lines = []
        for k in range(3):
            pname = CIFAR100_CLASSES[int(top5[idx, k])]
            pprob = float(probs[idx, k])
            marker = "✓" if int(top5[idx, k]) == int(true[idx]) else " "
            lines.append(f"{marker} {_short_label(pname, 14):<14s} {pprob * 100:5.1f}%")
        ax.text(0.5, -0.04, "\n".join(lines),
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=7, family="monospace",
                color="#222222")

    # Trim any unused cells if picked < n_cells (shouldn't happen with 10000 imgs).
    for extra in range(len(picked), n_cells):
        r, c = divmod(extra, args.cols)
        axes[r, c].axis("off")

    acc_shown = np.mean(pred[picked] == true[picked])
    fig.suptitle(
        f"ViT-Small CIFAR-100 inference samples  "
        f"(green=correct, red=wrong; shown acc {acc_shown * 100:.0f}% "
        f"of {n_cells}, population mean {np.mean(correct_mask) * 100:.1f}%)",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if args.output is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(args.predictions_npz)),
            "inference_thumbnails.png",
        )
    else:
        output_path = os.path.abspath(args.output)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
