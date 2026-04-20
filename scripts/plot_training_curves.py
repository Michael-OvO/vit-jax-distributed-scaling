"""Plot train/test loss and accuracy over epochs.

Trainer emits ``epoch_metrics.csv`` with columns
``epoch,train_loss,train_accuracy,test_loss,test_accuracy``; this renders
them as two side-by-side subplots and saves to ``training_curves.png``
alongside the CSV.

Usage:
    python scripts/plot_training_curves.py outputs/.../epoch_metrics.csv
    python scripts/plot_training_curves.py outputs/.../epoch_metrics.csv --title "ViT-Small, 100 epochs"
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})
    return rows


def _argmin(values):
    return min(range(len(values)), key=lambda i: values[i])


def _argmax(values):
    return max(range(len(values)), key=lambda i: values[i])


def plot(csv_path: str, title: str, output_path: str) -> None:
    rows = _read_csv(csv_path)
    if not rows:
        raise SystemExit(f"No rows in {csv_path}")

    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [r["train_loss"] for r in rows]
    test_loss = [r.get("test_loss", float("nan")) for r in rows]
    train_acc = [r["train_accuracy"] for r in rows]
    test_acc = [r.get("test_accuracy", float("nan")) for r in rows]

    min_test_loss_epoch = epochs[_argmin(test_loss)]
    peak_test_acc_epoch = epochs[_argmax(test_acc)]
    peak_test_acc = test_acc[_argmax(test_acc)]
    final_gap = train_acc[-1] - test_acc[-1]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ------- Loss -------
    ax_loss.plot(epochs, train_loss, "-", color="#1f77b4", linewidth=2, label="Train")
    ax_loss.plot(epochs, test_loss, "-", color="#d62728", linewidth=2, label="Test")
    ax_loss.axvline(min_test_loss_epoch, color="gray", linestyle="--", linewidth=1,
                    label=f"Test-loss min (epoch {min_test_loss_epoch})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right")

    # ------- Accuracy -------
    ax_acc.plot(epochs, [100 * a for a in train_acc], "-", color="#1f77b4",
                linewidth=2, label="Train")
    ax_acc.plot(epochs, [100 * a for a in test_acc], "-", color="#d62728",
                linewidth=2, label="Test")
    ax_acc.axhline(100 * peak_test_acc, color="gray", linestyle=":", linewidth=1,
                   label=f"Peak test {100*peak_test_acc:.1f}% (epoch {peak_test_acc_epoch})")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.set_title(f"Accuracy  (final train-test gap: {100*final_gap:.1f} pts)")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim(0, 105)
    ax_acc.legend(loc="lower right")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"  Min test loss:  {min(test_loss):.4f}  (epoch {min_test_loss_epoch})")
    print(f"  Peak test acc:  {100 * peak_test_acc:.2f}%  (epoch {peak_test_acc_epoch})")
    print(f"  Final test acc: {100 * test_acc[-1]:.2f}%")
    print(f"  Final train-test gap: {100 * final_gap:.2f} pts")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to epoch_metrics.csv")
    parser.add_argument("--title", default="", help="Figure title (optional)")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: training_curves.png next to the CSV)")
    args = parser.parse_args()

    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                         "training_curves.png")
    plot(args.csv, args.title, output)


if __name__ == "__main__":
    main()
