"""pmap-based distributed data parallelism for JAX.

Provides:
- create_train_step / create_train_step_with_straggler: pmap'd training steps
- create_eval_step: pmap'd evaluation step
- replicate_state / unreplicate_state: device replication helpers
- create_train_state: initialise TrainState with AdamW + cosine schedule
"""

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import jax_utils
from flax.training import train_state


TrainState = train_state.TrainState


def _loss_fn(params, apply_fn, batch, train=True, rng=None):
    """Compute cross-entropy loss and accuracy."""
    images, labels = batch["image"], batch["label"]
    rngs = {"dropout": rng} if rng is not None else None
    logits = apply_fn({"params": params}, images, train=train, rngs=rngs)
    loss = jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    )
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, accuracy


def create_train_step(devices=None):
    """Return a pmap'd training step function.

    Args:
        devices: Optional explicit list of devices to map across. If ``None``,
            `pmap` uses all local devices. Needed for the scaling experiment
            which benchmarks subsets of devices on multi-device hosts (TPU v3-8,
            multi-GPU VMs, etc.).

    Returns:
        A function (state, batch, rng) -> (new_state, metrics_dict),
        mapped across devices with axis_name='batch'.
    """

    @partial(jax.pmap, axis_name="batch", devices=devices)
    def train_step(state, batch, rng):
        rng, dropout_rng = jax.random.split(rng)

        def loss_wrapper(params):
            return _loss_fn(params, state.apply_fn, batch, train=True, rng=dropout_rng)

        grad_fn = jax.value_and_grad(loss_wrapper, has_aux=True)
        (loss, accuracy), grads = grad_fn(state.params)

        # Average gradients across devices (the core all-reduce operation).
        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")
        accuracy = jax.lax.pmean(accuracy, axis_name="batch")

        new_state = state.apply_gradients(grads=grads)
        metrics = {"loss": loss, "accuracy": accuracy}
        return new_state, metrics

    return train_step


def create_train_step_with_straggler(delay_iterations=1000, devices=None):
    """Return a pmap'd training step that injects real compute delay on device 0.

    On device index 0, a ``fori_loop`` executes ``delay_iterations`` nonlinear
    matrix operations that XLA cannot algebraically elide. Because pmap
    synchronises at the all-reduce barrier, device 0's extra work forces every
    other device to wait — this is the signature of synchronous-SGD's worst case.

    Design choices, all necessary on modern accelerators (v6e, H100):

    * **Nonlinear body.** ``tanh(a @ a) * 0.99 + 0.01 * a`` — each iteration
      truly depends on the previous, and the nonlinearity blocks the
      algebraic-simplification pass. The earlier ``a @ eye(64) + 0*a`` was
      folded to the identity map.
    * **Matrix size 512.** ``64 x 64`` was ~0.5 MFLOPs/iter; on a v6e this is
      ~5 nanoseconds and gets lost in noise. ``512 x 512`` is ~268 MFLOPs
      (~3 microseconds), so even moderate ``delay_iterations`` produce
      millisecond-scale delays visible against a ~50 ms step.
    * **Keep-alive via optimization_barrier.** The loop output is threaded
      into ``loss`` with a ``1e-30`` coupling. The coefficient is numerically
      negligible (below any training-relevant epsilon) but not an exact-zero
      identity, so XLA's DCE cannot prune the ``fori_loop``.

    Args:
        delay_iterations: Trip count of the delay loop on device 0.
        devices: Optional explicit device list; see :func:`create_train_step`.
    """

    @partial(jax.pmap, axis_name="batch", devices=devices)
    def train_step(state, batch, rng):
        rng, dropout_rng = jax.random.split(rng)

        def loss_wrapper(params):
            return _loss_fn(params, state.apply_fn, batch, train=True, rng=dropout_rng)

        grad_fn = jax.value_and_grad(loss_wrapper, has_aux=True)
        (loss, accuracy), grads = grad_fn(state.params)

        grads = jax.lax.pmean(grads, axis_name="batch")
        loss = jax.lax.pmean(loss, axis_name="batch")
        accuracy = jax.lax.pmean(accuracy, axis_name="batch")

        new_state = state.apply_gradients(grads=grads)

        # ---- Straggler injection on device 0 ----------------------------
        device_idx = jax.lax.axis_index("batch")
        side = 512

        def _straggler(acc):
            def body(i, a):
                return jnp.tanh(a @ a) * 0.99 + 0.01 * a
            return jax.lax.fori_loop(0, delay_iterations, body, acc)

        def _no_delay(acc):
            return acc

        # Non-uniform initialiser so early iterations do real work
        # (a @ a of a constant tensor would still get folded).
        anchor = jnp.linspace(-1.0, 1.0, side * side,
                              dtype=jnp.float32).reshape(side, side)
        delay_out = jax.lax.cond(device_idx == 0, _straggler, _no_delay, anchor)

        # Keep the fori_loop alive through XLA's DCE. The optimization_barrier
        # prevents reasoning across the line; the 1e-30 coupling is
        # numerically inert but not an exact-zero identity.
        loss, delay_sum = jax.lax.optimization_barrier(
            (loss, jnp.sum(delay_out)))
        loss = loss + 1e-30 * delay_sum

        metrics = {"loss": loss, "accuracy": accuracy}
        return new_state, metrics

    return train_step


def create_eval_step(devices=None):
    """Return a pmap'd evaluation step function.

    Args:
        devices: Optional explicit device list; see :func:`create_train_step`.

    Returns:
        A function (state, batch) -> metrics_dict,
        mapped across devices with axis_name='batch'.
    """

    @partial(jax.pmap, axis_name="batch", devices=devices)
    def eval_step(state, batch):
        loss, accuracy = _loss_fn(
            state.params, state.apply_fn, batch, train=False
        )
        loss = jax.lax.pmean(loss, axis_name="batch")
        accuracy = jax.lax.pmean(accuracy, axis_name="batch")
        return {"loss": loss, "accuracy": accuracy}

    return eval_step


def replicate_state(state, devices=None):
    """Replicate a TrainState across the given (or all local) devices."""
    return jax_utils.replicate(state, devices=devices)


def unreplicate_state(state):
    """Retrieve the TrainState from the first device."""
    return jax_utils.unreplicate(state)


def create_train_state(
    rng,
    model,
    learning_rate,
    image_size=32,
    weight_decay=0.01,
    warmup_steps=500,
    total_steps=10000,
):
    """Create a TrainState with AdamW and cosine decay + warmup schedule.

    Args:
        rng: JAX PRNG key for parameter initialisation.
        model: A Flax module (e.g. ViTSmall).
        learning_rate: Peak learning rate after warmup.
        image_size: Spatial resolution of input images.
        weight_decay: AdamW weight decay coefficient.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total training steps (warmup + cosine decay).

    Returns:
        A ``TrainState`` ready to be replicated across devices.
    """
    dummy_input = jnp.ones([1, image_size, image_size, 3])
    variables = model.init(rng, dummy_input, train=False)
    params = variables["params"]

    # Optax subtracts warmup from decay_steps internally; clamp so short runs
    # (e.g. 2-epoch smoke tests with fewer total steps than the default 500
    # warmup) don't hit ValueError on a negative decay horizon.
    effective_warmup = min(warmup_steps, max(1, total_steps // 2))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=effective_warmup,
        decay_steps=total_steps,
        end_value=0.0,
    )

    optimizer = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
    )
