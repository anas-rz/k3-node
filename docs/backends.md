# Multi-Backend Guide

K3 Node runs natively across **TensorFlow**, **PyTorch**, and **JAX** via Keras 3.

---

## Selecting a Backend

You can select the active backend using the `KERAS_BACKEND` environment variable before importing `keras` or `k3_node`:

=== "Linux / macOS (Bash)"
    ```bash
    # PyTorch
    export KERAS_BACKEND=torch
    python train.py

    # TensorFlow
    export KERAS_BACKEND=tensorflow
    python train.py

    # JAX
    export KERAS_BACKEND=jax
    python train.py
    ```

=== "Python Inline"
    ```python
    import os
    os.environ["KERAS_BACKEND"] = "torch"  # Must be set before importing keras

    import keras
    import k3_node
    print("Active backend:", keras.config.backend())
    ```

---

## Backend Comparison

| Feature | PyTorch (`torch`) | TensorFlow (`tensorflow`) | JAX (`jax`) |
|:---|:---|:---|:---|
| **Ecosystem** | PyTorch Geometric, HuggingFace | TensorFlow, Keras ecosystem | JAX, Flax, Optax |
| **GPU Support** | CUDA, ROCm, MPS (Apple Silicon) | CUDA, TPU | CUDA, TPU |
| **Graph Operations** | Native tensor indexing & scatter | `tf.gather`, `tf.math.segment_sum` | `jax.numpy`, `jax.lax.scatter` |
| **JIT Compilation** | `torch.compile` | `tf.function` | `jax.jit` (XLA) |

---

## Best Practices for Writing Backend-Agnostic GNNs

1. **Use `keras.ops` instead of backend-specific tensor libraries**:
   ```python
   # Recommended (Multi-backend)
   from keras import ops
   summed = ops.sum(x, axis=1)
   normed = ops.relu(x)

   # Avoid (Backend-specific)
   import torch
   summed = torch.sum(x, dim=1)
   ```

2. **Use `k3_node.layers.conv.MessagePassing`**:
   All message-passing logic, neighborhood scatter, and segment aggregations are automatically mapped to optimal multi-backend primitives.

3. **Data conversions**:
   Use `ops.convert_to_tensor` to prepare arrays for model ingestion:
   ```python
   from keras import ops
   tensor_x = ops.convert_to_tensor(numpy_array, dtype="float32")
   ```

