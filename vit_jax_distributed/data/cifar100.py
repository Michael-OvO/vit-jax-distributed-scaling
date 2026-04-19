"""CIFAR-100 data pipeline for JAX distributed training.

Loads CIFAR-100 via tensorflow_datasets, then operates entirely in numpy
for augmentation and batching. The full dataset fits comfortably in RAM
(~180 MB for images + labels).
"""

from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np

# CIFAR-100 channel-wise statistics (computed over training set, in [0, 1] scale).
CIFAR100_MEAN = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
CIFAR100_STD = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)

_NUM_TRAIN = 50_000
_NUM_TEST = 10_000
_NUM_CLASSES = 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_datasets(
    config: Any,
    *,
    num_devices: int = 1,
    seed: int = 0,
) -> Tuple[Iterator[Dict[str, np.ndarray]], Iterator[Dict[str, np.ndarray]]]:
    """Return train and test batch iterators for CIFAR-100.

    Parameters
    ----------
    config : object
        Must expose the following attributes:
            - ``batch_size`` (int): global batch size (across all devices).
            - ``image_size`` (int, optional): spatial resolution. Default 32.
            - ``data_augmentation`` (bool, optional): enable training augmentations.
              Default ``True``.
    num_devices : int
        Number of devices for ``pmap``-style sharding.  The effective batch
        size is rounded down to be divisible by ``num_devices``.
    seed : int
        Random seed for reproducible shuffling / augmentations.

    Returns
    -------
    train_iter, test_iter : iterators
        Each yields ``{'image': np.ndarray, 'label': np.ndarray}`` batches.
        Training iterator reshuffles every epoch and loops forever.
        Test iterator yields one full pass then stops.
    """
    batch_size = config.batch_size
    image_size = getattr(config, "image_size", 32)
    augment = getattr(config, "data_augmentation", True)

    # Make batch size divisible by num_devices (round down).
    batch_size = (batch_size // num_devices) * num_devices

    train_images, train_labels = _load_split("train", image_size)
    test_images, test_labels = _load_split("test", image_size)

    train_iter = _numpy_data_iterator(
        train_images,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        augment=augment,
        loop=True,
        seed=seed,
    )
    test_iter = _numpy_data_iterator(
        test_images,
        test_labels,
        batch_size=batch_size,
        shuffle=False,
        augment=False,
        loop=False,
        seed=seed,
    )
    return train_iter, test_iter


def shard_batch(
    batch: Dict[str, np.ndarray],
    num_devices: int,
) -> Dict[str, np.ndarray]:
    """Reshape batch arrays from ``(B, ...)`` to ``(num_devices, B // num_devices, ...)``.

    Raises ``ValueError`` if the batch dimension is not divisible by
    ``num_devices``.
    """
    def _reshape(x: np.ndarray) -> np.ndarray:
        if x.shape[0] % num_devices != 0:
            raise ValueError(
                f"Batch dimension {x.shape[0]} is not divisible by "
                f"num_devices={num_devices}."
            )
        per_device = x.shape[0] // num_devices
        return x.reshape((num_devices, per_device) + x.shape[1:])

    return {k: _reshape(v) for k, v in batch.items()}


def get_num_examples(split: str) -> int:
    """Return the number of examples in the given CIFAR-100 split.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    """
    if split == "train":
        return _NUM_TRAIN
    elif split == "test":
        return _NUM_TEST
    else:
        raise ValueError(f"Unknown split: {split!r}. Expected 'train' or 'test'.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SPLIT_CACHE: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}


def _load_split(
    split: str,
    image_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CIFAR-100 split via TFDS and return numpy arrays.

    Results are cached so repeated calls (e.g. creating test iterators
    each epoch) do not re-read from disk.

    Returns
    -------
    images : np.ndarray, float32, shape ``(N, H, W, 3)``
    labels : np.ndarray, int32, shape ``(N,)``
    """
    cache_key = (split, image_size)
    if cache_key in _SPLIT_CACHE:
        return _SPLIT_CACHE[cache_key]

    import tensorflow_datasets as tfds

    ds = tfds.load(
        "cifar100",
        split=split,
        as_supervised=True,
        batch_size=-1,  # load everything at once
    )
    images, labels = tfds.as_numpy(ds)

    # uint8 -> float32 in [0, 1]
    images = images.astype(np.float32) / 255.0

    # Resize if needed (rare for CIFAR-100 but supported).
    if image_size != 32:
        images = _resize_images(images, image_size)

    # Standardise with channel-wise mean / std.
    images = (images - CIFAR100_MEAN) / CIFAR100_STD

    labels = labels.astype(np.int32)
    _SPLIT_CACHE[cache_key] = (images, labels)
    return images, labels


def _resize_images(images: np.ndarray, size: int) -> np.ndarray:
    """Resize images to ``(size, size)`` using nearest-neighbour (numpy only).

    Good enough for small upscales/downscales on CIFAR-sized images.  For
    higher-fidelity resizing, use a dedicated library.
    """
    n, h_in, w_in, c = images.shape
    if h_in == size and w_in == size:
        return images

    row_idx = (np.arange(size) * h_in / size).astype(np.int32)
    col_idx = (np.arange(size) * w_in / size).astype(np.int32)
    return images[:, row_idx[:, None], col_idx[None, :], :]


def _numpy_data_iterator(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    augment: bool,
    loop: bool,
    seed: int,
) -> Iterator[Dict[str, np.ndarray]]:
    """Yield batches of ``{'image': ..., 'label': ...}`` from in-memory arrays.

    Parameters
    ----------
    images, labels : np.ndarray
        Full dataset arrays.
    batch_size : int
        Must already be divisible by ``num_devices`` if sharding is desired.
    shuffle : bool
        Shuffle indices at the start of each epoch (training).
    augment : bool
        Apply random horizontal flip + random crop w/ padding.
    loop : bool
        If ``True``, iterate forever (training).  Otherwise yield one epoch.
    seed : int
        Base seed for the RNG.
    """
    n = images.shape[0]
    rng = np.random.RandomState(seed)

    # Truncate to largest multiple of batch_size so every batch is full.
    usable = (n // batch_size) * batch_size

    while True:
        indices = np.arange(n)
        if shuffle:
            rng.shuffle(indices)
        indices = indices[:usable]

        for start in range(0, usable, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_images = images[batch_idx].copy()
            batch_labels = labels[batch_idx]

            if augment:
                batch_images = _augment_batch(batch_images, rng)

            yield {"image": batch_images, "label": batch_labels}

        if not loop:
            break


# ---------------------------------------------------------------------------
# Numpy augmentations
# ---------------------------------------------------------------------------

def _augment_batch(
    images: np.ndarray,
    rng: np.random.RandomState,
    pad: int = 4,
) -> np.ndarray:
    """Apply random horizontal flip and random crop with padding.

    Parameters
    ----------
    images : np.ndarray, shape ``(B, H, W, C)``
    rng : numpy RandomState
    pad : int
        Padding pixels on each side before random crop.

    Returns
    -------
    Augmented images with the same shape as input.
    """
    b, h, w, c = images.shape

    # --- Random horizontal flip (per-image) ---
    flip_mask = rng.rand(b) < 0.5
    images[flip_mask] = images[flip_mask, :, ::-1, :]

    # --- Random crop with reflection padding ---
    padded = np.pad(
        images,
        ((0, 0), (pad, pad), (pad, pad), (0, 0)),
        mode="reflect",
    )

    # Sample random offsets for each image in the batch.
    crop_y = rng.randint(0, 2 * pad + 1, size=b)
    crop_x = rng.randint(0, 2 * pad + 1, size=b)

    # Vectorized crop using advanced indexing (avoids per-image Python loop).
    batch_idx = np.arange(b)[:, None, None]
    y_idx = np.arange(h)[None, :, None] + crop_y[:, None, None]
    x_idx = np.arange(w)[None, None, :] + crop_x[:, None, None]
    cropped = padded[batch_idx, y_idx, x_idx, :]

    return cropped
