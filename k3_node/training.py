"""Backend-agnostic training helpers for models that don't fit ``keras.Model.fit``.

Examples are models with several optimizers (like the adversarial autoencoders) or losses that
are computed outside of a model's ``call``.
"""
from typing import Callable, Sequence

import keras
from keras import ops


def _seed_state():
    from keras.src.random.seed_generator import global_seed_generator

    return global_seed_generator().state


def gradient_step(loss_fn: Callable, variables: Sequence, optimizer, state_variables: Sequence = ()):
    r"""Runs ``loss_fn()``, then updates ``variables`` with one ``optimizer`` step on its gradients.

    Works eagerly on every backend. Variables used by ``loss_fn`` but not listed in ``variables``
    are not trained. ``state_variables`` are non-trainable variables that ``loss_fn`` may update
    (for example the moving statistics of batch normalization); this matters on JAX only.

    Args:
        loss_fn (callable): A function without arguments that returns a scalar loss.
        variables (list): The variables to train, e.g. ``model.trainable_variables``.
        optimizer (keras.optimizers.Optimizer): The optimizer applying the update.
        state_variables (list, optional): Non-trainable variables updated by ``loss_fn``.

    Returns:
        The loss as a Python float.

    Example:
        ```python
        import keras
        from keras import ops
        from k3_node.training import gradient_step

        w = keras.Variable(3.0)
        optimizer = keras.optimizers.SGD(learning_rate=0.25)
        loss = gradient_step(lambda: ops.square(w), [w], optimizer)  # gradient 2 * w = 6
        print(loss, float(ops.convert_to_numpy(w)))  # 9.0 1.5
        ```
    """
    variables = list(variables)
    backend = keras.config.backend()

    if backend == "tensorflow":
        import tensorflow as tf

        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, variables)
    elif backend == "torch":
        import torch

        loss = loss_fn()
        grads = torch.autograd.grad(loss, [v.value for v in variables], allow_unused=True)
    elif backend == "jax":
        import jax
        from keras.src.backend.common.stateless_scope import StatelessScope

        tracked = list(state_variables) + [_seed_state()]

        def compute(values):
            with StatelessScope(state_mapping=list(zip(variables, values))) as scope:
                out = loss_fn()
            return out, [scope.get_current_value(v) for v in tracked]

        (loss, updates), grads = jax.value_and_grad(compute, has_aux=True)([v.value for v in variables])
        for v, value in zip(tracked, updates):
            if value is not None:
                v.assign(value)
    else:
        raise NotImplementedError(f"gradient_step does not support the {backend} backend.")

    grads = [ops.zeros_like(v) if g is None else g for g, v in zip(grads, variables)]
    optimizer.apply_gradients(zip(grads, variables))
    return float(ops.convert_to_numpy(loss))
