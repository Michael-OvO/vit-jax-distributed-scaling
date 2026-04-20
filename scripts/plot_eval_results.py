"""Generate per-class accuracy bar chart and confusion matrix heatmap.

Input: the ``test_predictions.npz`` produced by ``full_test_eval.py``.
Output: two PNGs written next to the NPZ (or to --output_dir).
"""

from __future__ import annotations

import argparse
import os

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


def plot_per_class(acc: np.ndarray, output_path: str, overall_acc: float,
                   top5: float) -> None:
    order = np.argsort(acc)                         # worst first
    fig, ax = plt.subplots(figsize=(10, 18))
    y = np.arange(100)
    colors = plt.cm.RdYlGn(acc[order])              # red low, green high
    ax.barh(y, acc[order] * 100, color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels([CIFAR100_CLASSES[i] for i in order], fontsize=7)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title(
        f"Per-class test accuracy  (top-1 mean {overall_acc * 100:.1f}%, "
        f"top-5 {top5 * 100:.1f}%)"
    )
    ax.axvline(overall_acc * 100, color="black", linestyle="--", linewidth=1.2,
               label=f"Mean {overall_acc * 100:.1f}%")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 99.5)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_confusion(confusion: np.ndarray, output_path: str) -> None:
    cm = confusion.astype(float)
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(14, 13))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="equal",
                   interpolation="nearest")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (rows sum to 1; perfect model = pure diagonal)")
    ax.set_xticks(np.arange(100))
    ax.set_xticklabels(CIFAR100_CLASSES, rotation=90, fontsize=5)
    ax.set_yticks(np.arange(100))
    ax.set_yticklabels(CIFAR100_CLASSES, fontsize=5)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="P(predicted | true)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved {output_path}")


def _top_confusions(confusion: np.ndarray, k: int = 10):
    """Return (true, predicted, count) for the top k off-diagonal entries."""
    cm = confusion.copy()
    np.fill_diagonal(cm, 0)
    flat = np.argsort(cm, axis=None)[::-1][:k]
    pairs = []
    for idx in flat:
        t, p = np.unravel_index(idx, cm.shape)
        if cm[t, p] == 0:
            break
        pairs.append((int(t), int(p), int(cm[t, p])))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_npz")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.dirname(
        os.path.abspath(args.predictions_npz)
    )
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(args.predictions_npz)
    true = data["true_labels"]
    pred = data["pred_labels"]
    top5 = data["top5_indices"]
    per_class = data["per_class_acc"]
    confusion = data["confusion"]

    top1_acc = float(np.mean(pred == true))
    top5_acc = float(np.mean(np.any(top5 == true[:, None], axis=1)))

    print(f"Top-1 accuracy: {top1_acc * 100:.2f}%")
    print(f"Top-5 accuracy: {top5_acc * 100:.2f}%")

    best5 = np.argsort(per_class)[::-1][:5]
    worst5 = np.argsort(per_class)[:5]
    print("\nBest 5 classes:")
    for i in best5:
        print(f"  {CIFAR100_CLASSES[i]:18s}  {per_class[i] * 100:5.1f}%")
    print("Worst 5 classes:")
    for i in worst5:
        print(f"  {CIFAR100_CLASSES[i]:18s}  {per_class[i] * 100:5.1f}%")

    print("\nMost common confusions (true -> predicted):")
    for t, p, c in _top_confusions(confusion, 10):
        print(f"  {CIFAR100_CLASSES[t]:16s} -> {CIFAR100_CLASSES[p]:16s}  {c}x")

    plot_per_class(per_class, os.path.join(output_dir, "per_class_accuracy.png"),
                   top1_acc, top5_acc)
    plot_confusion(confusion, os.path.join(output_dir, "confusion_matrix.png"))


if __name__ == "__main__":
    main()
