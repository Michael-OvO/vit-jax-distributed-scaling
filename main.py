"""Entry point for JAX distributed ViT training experiments."""

import jax

from vit_jax_distributed.utils.config import get_config
from vit_jax_distributed.train.trainer import train
from vit_jax_distributed.experiments.scaling import run_scaling_experiment, ScalingConfig
from vit_jax_distributed.experiments.straggler import run_straggler_experiment, StragglerConfig


def _apply_precision(precision: str) -> None:
    """Globally enable bf16 matmuls when requested.

    TPU MXUs execute matmuls natively in bf16; setting the default precision
    keeps fp32 params/optimizer state but casts inside every `@`, `einsum`,
    Conv, and Dense. fp32 is the JAX default (no-op).
    """
    if precision == "bf16":
        jax.config.update("jax_default_matmul_precision", "bfloat16")


def _base_experiment_kwargs(config):
    """Extract common experiment fields from argparse Namespace."""
    return {
        "batch_size": config.batch_size,
        "image_size": config.image_size,
        "data_augmentation": config.data_augmentation,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps,
        "output_dir": config.output_dir,
        "seed": config.seed,
        "num_devices": config.num_devices if config.num_devices > 0 else None,
    }


def main():
    config = get_config()
    _apply_precision(config.precision)

    print(f"JAX devices: {jax.devices()}")
    print(f"Local device count: {jax.local_device_count()}")
    print(f"Matmul precision: {config.precision}")

    if config.experiment == "train":
        train(config)
    elif config.experiment == "scaling":
        run_scaling_experiment(ScalingConfig(**_base_experiment_kwargs(config)))
    elif config.experiment == "straggler":
        kwargs = _base_experiment_kwargs(config)
        kwargs["straggler_delay"] = config.straggler_delay
        run_straggler_experiment(StragglerConfig(**kwargs))
    else:
        raise ValueError(f"Unknown experiment: {config.experiment}")


if __name__ == "__main__":
    main()
