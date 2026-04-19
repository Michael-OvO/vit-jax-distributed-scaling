"""CLI configuration for distributed ViT training experiments."""

import argparse


def get_config():
    """Parse command-line arguments and return an ``argparse.Namespace``."""
    parser = argparse.ArgumentParser(
        description="JAX Distributed ViT Training"
    )

    # Training hyper-parameters
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Model architecture
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--mlp_dim", type=int, default=768)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # Experiment settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["train", "scaling", "straggler"],
        default="train",
    )
    parser.add_argument("--num_devices", type=int, default=0,
                        help="Number of devices (0 = use all available)")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="fp32",
    )
    parser.add_argument("--straggler_delay", type=int, default=1000,
                        help="Iterations of dummy work for straggler simulation")

    # Logging / evaluation
    parser.add_argument("--log_every", type=int, default=50,
                        help="Steps between logging")
    parser.add_argument("--eval_every", type=int, default=1,
                        help="Epochs between evaluation")
    parser.add_argument("--data_augmentation", action="store_true",
                        default=True,
                        help="Enable training data augmentation (default: True)")
    parser.add_argument("--no_data_augmentation", action="store_false",
                        dest="data_augmentation",
                        help="Disable training data augmentation")

    return parser.parse_args()


def config_to_dict(config):
    """Convert an ``argparse.Namespace`` to a plain ``dict``."""
    return vars(config)
