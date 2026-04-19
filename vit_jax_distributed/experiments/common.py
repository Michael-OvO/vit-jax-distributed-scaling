"""Shared configuration and helpers for experiment modules."""

from dataclasses import dataclass
from typing import Optional


class SimpleDataConfig:
    """Lightweight config object accepted by ``get_datasets``."""

    def __init__(self, batch_size: int, image_size: int = 32, data_augmentation: bool = True):
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_augmentation = data_augmentation


@dataclass
class ExperimentConfig:
    """Base configuration shared by all benchmark experiments."""

    batch_size: int = 256
    image_size: int = 32
    data_augmentation: bool = True
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    total_steps: int = 10000
    benchmark_steps: int = 100
    warmup_benchmark_steps: int = 10
    output_dir: str = "results"
    seed: int = 42
    num_devices: Optional[int] = None
