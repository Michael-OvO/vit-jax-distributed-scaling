"""Vision Transformer (ViT-Small) for CIFAR-100 in Flax Linen.

A complete ViT-Small implementation optimized for 32x32 CIFAR images with
small patch sizes and configurable depth.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional


class PatchEmbedding(nn.Module):
    """Splits an image into patches and projects them to the hidden dimension.

    Uses a single Conv layer with kernel_size=patch_size and stride=patch_size
    to extract non-overlapping patches, then adds a learnable CLS token and
    positional embeddings.
    """

    hidden_dim: int = 384
    patch_size: int = 4
    image_size: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size = x.shape[0]
        num_patches = (self.image_size // self.patch_size) ** 2

        # Project patches: (B, H, W, C) -> (B, num_patches, hidden_dim)
        x = nn.Dense(
            features=self.hidden_dim,
            name="patch_projection",
        )(
            # Reshape image into flattened patches first
            x.reshape(
                batch_size,
                self.image_size // self.patch_size,
                self.patch_size,
                self.image_size // self.patch_size,
                self.patch_size,
                -1,
            )
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(batch_size, num_patches, -1)
        )

        # Learnable CLS token: (1, 1, hidden_dim)
        cls_token = self.param(
            "cls_token",
            nn.initializers.normal(stddev=0.02),
            (1, 1, self.hidden_dim),
        )
        cls_tokens = jnp.broadcast_to(cls_token, (batch_size, 1, self.hidden_dim))

        # Prepend CLS token: (B, num_patches+1, hidden_dim)
        x = jnp.concatenate([cls_tokens, x], axis=1)

        # Learnable positional embeddings for all tokens (CLS + patches)
        pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, num_patches + 1, self.hidden_dim),
        )
        x = x + pos_embedding

        return x


class MLP(nn.Module):
    """Two-layer MLP with GELU activation and dropout."""

    hidden_dim: int = 384
    mlp_dim: int = 768
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN -> MHSA -> residual -> LN -> MLP -> residual.

    Uses pre-LayerNorm (applying LayerNorm before attention/MLP) which is the
    standard for ViT and improves training stability.
    """

    hidden_dim: int = 384
    num_heads: int = 6
    mlp_dim: int = 768
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        # Pre-norm multi-head self-attention with residual
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x, x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = x + residual

        # Pre-norm MLP with residual
        residual = x
        x = nn.LayerNorm()(x)
        x = MLP(
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            dropout_rate=self.dropout_rate,
        )(x, deterministic=deterministic)
        x = x + residual

        return x


class ViTSmall(nn.Module):
    """Vision Transformer Small for CIFAR-100.

    Architecture: PatchEmbedding -> N x TransformerBlock -> LayerNorm -> CLS head.

    Designed for 32x32 CIFAR images with patch_size=4 yielding 8x8=64 patches,
    keeping the sequence length manageable while preserving spatial detail.

    Attributes:
        num_classes: Number of output classes (100 for CIFAR-100).
        patch_size: Size of each image patch (4 for 32x32 images).
        hidden_dim: Transformer hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks (default 8 for speed).
        mlp_dim: Hidden dimension of the feed-forward MLP.
        dropout_rate: Dropout probability for attention and MLP layers.
        image_size: Input image spatial resolution.
    """

    num_classes: int = 100
    patch_size: int = 4
    hidden_dim: int = 384
    num_heads: int = 6
    num_layers: int = 8
    mlp_dim: int = 768
    dropout_rate: float = 0.1
    image_size: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input images of shape (batch, height, width, channels).
            train: Whether the model is in training mode (enables dropout).

        Returns:
            Classification logits of shape (batch, num_classes).
        """
        deterministic = not train

        # Patch embedding with CLS token and positional embeddings
        x = PatchEmbedding(
            hidden_dim=self.hidden_dim,
            patch_size=self.patch_size,
            image_size=self.image_size,
        )(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Transformer encoder blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
            )(x, deterministic=deterministic)

        # Final LayerNorm
        x = nn.LayerNorm()(x)

        # CLS token pooling: extract the first token
        x = x[:, 0]

        # Classification head
        x = nn.Dense(self.num_classes)(x)

        return x
