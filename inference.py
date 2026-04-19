"""Classify a single image with a trained ViT-Small checkpoint.

Usage
-----

Classify a random CIFAR-100 test image (ground-truth known):

    python inference.py --checkpoint outputs/tpu_train/checkpoint_latest.msgpack

Classify your own image:

    python inference.py \
        --checkpoint outputs/tpu_train/checkpoint_latest.msgpack \
        --image my_cat.jpg

The script reconstructs the model architecture from the metadata that
`save_replicated` wrote next to the checkpoint, so you don't have to re-pass
`--num_layers`, `--hidden_dim`, etc. --- they come from the training run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from vit_jax_distributed.data.cifar100 import (
    CIFAR100_MEAN,
    CIFAR100_STD,
    _load_split,
)
from vit_jax_distributed.distributed.parallel import create_train_state
from vit_jax_distributed.models.vit import ViTSmall
from vit_jax_distributed.utils.checkpoint import load_checkpoint


# 100 CIFAR-100 fine class names, alphabetical (the order used by tfds).
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


def _load_metadata(checkpoint_path: str) -> dict:
    """Try to read the adjacent ``checkpoint_latest.json`` for hyper-params."""
    directory = os.path.dirname(os.path.abspath(checkpoint_path))
    for candidate in ("checkpoint_latest.json",):
        meta_path = os.path.join(directory, candidate)
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
    return {}


def _preprocess_image(path: str, image_size: int) -> np.ndarray:
    """Load an arbitrary image file, resize, standardise to CIFAR-100 stats."""
    try:
        from PIL import Image
    except ImportError as e:
        raise SystemExit(
            "Pillow is required for --image. Install with `pip install Pillow`."
        ) from e
    img = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return (arr - CIFAR100_MEAN) / CIFAR100_STD


def _sample_cifar_test_image(seed: int) -> Tuple[np.ndarray, int]:
    """Return one random CIFAR-100 test image + its true label index."""
    images, labels = _load_split("test")
    rng = np.random.RandomState(seed)
    idx = rng.randint(len(images))
    return images[idx], int(labels[idx])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained ViT-Small on one image."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to a .msgpack checkpoint (from save_replicated)")
    parser.add_argument("--image", default=None,
                        help="Image file (jpg/png). If omitted, samples a random "
                             "CIFAR-100 test image (with ground truth).")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed when sampling a CIFAR test image.")

    # Model hyper-parameters. Pulled from checkpoint metadata by default; use
    # these overrides only if metadata is missing or the checkpoint was
    # produced with custom config outside the main trainer.
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--mlp_dim", type=int, default=None)
    args = parser.parse_args()

    # ---- Resolve model hyper-parameters ---------------------------------
    meta = _load_metadata(args.checkpoint)
    cfg = meta.get("config", {})

    def resolve(attr, fallback):
        cli = getattr(args, attr)
        return cli if cli is not None else cfg.get(attr, fallback)

    image_size = resolve("image_size", 32)
    patch_size = resolve("patch_size", 4)
    num_layers = resolve("num_layers", 8)
    hidden_dim = resolve("hidden_dim", 384)
    num_heads = resolve("num_heads", 6)
    mlp_dim = resolve("mlp_dim", 768)

    print(f"Model: hidden={hidden_dim}, layers={num_layers}, heads={num_heads}, "
          f"patch={patch_size}, image={image_size}")

    # ---- Build model + template state ------------------------------------
    model = ViTSmall(
        num_classes=100,
        patch_size=patch_size,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        mlp_dim=mlp_dim,
        dropout_rate=0.0,  # inference only
        image_size=image_size,
    )
    template = create_train_state(
        jax.random.PRNGKey(0),
        model,
        learning_rate=1e-3,
        image_size=image_size,
        total_steps=2,  # placeholder; schedule is irrelevant for inference
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    state = load_checkpoint(args.checkpoint, template)
    if meta:
        print(f"Trained for {meta.get('step', '?')} steps "
              f"(epoch {meta.get('epoch', '?')} of "
              f"{meta.get('total_epochs', '?')}).")

    # ---- Load image ------------------------------------------------------
    if args.image is not None:
        print(f"Image: {args.image}")
        image = _preprocess_image(args.image, image_size)
        true_label = None
    else:
        image, true_label = _sample_cifar_test_image(args.seed)
        print(f"No --image provided; sampled CIFAR-100 test image "
              f"(true class: {CIFAR100_CLASSES[true_label]!r}).")

    # ---- Forward pass ---------------------------------------------------
    logits = model.apply({"params": state.params}, image[None], train=False)
    probs = jax.nn.softmax(logits[0])
    probs_np = np.asarray(probs)

    order = np.argsort(-probs_np)
    k = min(args.topk, 100)
    print(f"\nTop-{k} predictions:")
    for rank, idx in enumerate(order[:k], start=1):
        marker = "  <-- true class" if true_label is not None and idx == true_label else ""
        print(f"  {rank:2d}. {CIFAR100_CLASSES[idx]:18s}  "
              f"p = {probs_np[idx]:.4f}{marker}")

    if true_label is not None:
        predicted = int(order[0])
        verdict = "CORRECT" if predicted == true_label else "WRONG"
        print(f"\n{verdict}: model predicts '{CIFAR100_CLASSES[predicted]}'; "
              f"truth is '{CIFAR100_CLASSES[true_label]}'.")


if __name__ == "__main__":
    main()
