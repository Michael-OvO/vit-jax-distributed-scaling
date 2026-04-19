"""Minimal TrainState checkpointing via flax.serialization.

Avoids a hard dependency on orbax/cloud-tpu-checkpoint; a TrainState
serialises cleanly to msgpack and restores against an identically-shaped
template.
"""

import json
import os
from typing import Optional

from flax import serialization

from vit_jax_distributed.distributed.parallel import unreplicate_state


LATEST_CHECKPOINT = "checkpoint_latest.msgpack"
LATEST_METADATA = "checkpoint_latest.json"


def save_checkpoint(
    state,
    output_dir: str,
    step: int,
    metadata: Optional[dict] = None,
    keep_numbered: bool = True,
) -> str:
    """Save a (non-replicated) TrainState to ``output_dir``.

    Writes two files:

    * ``checkpoint_latest.msgpack`` — always overwritten; the pointer inference
      scripts load by default.
    * ``checkpoint_<step>.msgpack`` — optional numbered history. Set
      ``keep_numbered=False`` on disk-constrained hosts; each copy is
      ~3x model-size bytes because AdamW holds two optimiser moments.

    Plus ``checkpoint_latest.json`` with ``{"step": ..., **metadata}`` so the
    inference side can reconstruct hyper-parameters without guessing.

    The caller must unreplicate the state first if training with pmap —
    serialising the replicated (N, ...) arrays would waste N x bytes and
    wouldn't restore cleanly.
    """
    os.makedirs(output_dir, exist_ok=True)

    payload = serialization.to_bytes(state)

    latest_path = os.path.join(output_dir, LATEST_CHECKPOINT)
    with open(latest_path, "wb") as f:
        f.write(payload)

    if keep_numbered:
        numbered_path = os.path.join(output_dir, f"checkpoint_{step:06d}.msgpack")
        with open(numbered_path, "wb") as f:
            f.write(payload)

    meta = {"step": int(step), **(metadata or {})}
    with open(os.path.join(output_dir, LATEST_METADATA), "w") as f:
        json.dump(meta, f, indent=2)

    return latest_path


def load_checkpoint(path: str, template_state):
    """Restore a TrainState from ``path`` using ``template_state`` as shape spec.

    ``template_state`` is a freshly-initialised TrainState with the same
    architecture and optimiser as was trained; only the parameter and moment
    *values* are overwritten.
    """
    with open(path, "rb") as f:
        return serialization.from_bytes(template_state, f.read())


def save_replicated(state, output_dir: str, step: int, **kwargs) -> str:
    """Convenience: unreplicate a pmap'd state then save.

    Equivalent to ``save_checkpoint(unreplicate_state(state), ...)``.
    """
    return save_checkpoint(unreplicate_state(state), output_dir, step, **kwargs)
