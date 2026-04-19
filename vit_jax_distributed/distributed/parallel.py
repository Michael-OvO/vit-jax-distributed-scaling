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
    """Return a pmap'd training step that injects artificial delay on device 0.

    On device index 0, a dummy fori_loop runs ``delay_iterations`` extra
    matrix multiplications to simulate a straggler worker. Because pmap
    synchronises at the all-reduce barrier, the straggler forces every
    device to wait — demonstrating the bottleneck of synchronous data
    parallelism.

    Args:
        delay_iterations: Number of dummy matmul iterations on device 0.
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

        # Inject artificial delay on device 0 via extra compute that XLA
        # cannot elide (each iteration depends on the previous result).
        device_idx = jax.lax.axis_index("batch")

        def _straggler_delay(dummy):
            def body_fn(i, acc):
                return acc @ jnp.eye(64) + 0.0 * acc
            return jax.lax.fori_loop(0, delay_iterations, body_fn, dummy)

        def _no_delay(dummy):
            return dummy

        dummy = jnp.ones((64, 64))
        _ = jax.lax.cond(device_idx == 0, _straggler_delay, _no_delay, dummy)

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
