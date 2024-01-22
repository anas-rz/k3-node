from keras import ops
from k3_node.layers.aggr import Aggregation


class SumAggregation(Aggregation):
    def call(self, x, index=None, axis=-2):
        return self.reduce(x, index, axis=axis, reduce_fn=ops.segment_sum)


class MaxAggregation(Aggregation):
    def call(self, x, index=None, axis=-2):
        return self.reduce(x, index, axis=axis, reduce_fn=ops.segment_max)


class MeanAggregation(Aggregation):
    def call(self, x, index=None, axis=-2):
        return self.reduce(x, index, axis=axis, reduce_fn=_segment_mean)


class SoftmaxAggregation(Aggregation):
    def __init__(self, t=1.0, trainable=False, channels=1):
        super().__init__()

        if not trainable and channels != 1:
            raise ValueError(
                "Cannot set 'channels' greater than '1' in case 'SoftmaxAggregation' is not trainable"
            )

        self._init_t = t
        self.trainable = trainable
        self.channels = channels

        self.t = self.add_weight((channels,), initializer="zeros") if trainable else t

    def call(self, x, index=None, axis=-2):
        t = self.t
        if self.channels != 1:
            self.assert_two_dimensional_input(x, axis)
            assert ops.is_tensor(t)
            t = ops.reshape(t, (1, self.channels))

        alpha = x
        if not isinstance(t, (int, float)) or t != 1:
            alpha = x * t
        alpha = ops.softmax(alpha, axis=axis)
        return self.reduce(x * alpha, index=index, axis=axis, reduce_fn=ops.segment_sum)


class PowerMeanAggregation(Aggregation):
    def __init__(self, p=1.0, trainable=False, channels=1):
        super().__init__()

        if not trainable and channels != 1:
            raise ValueError(
                f"Cannot set 'channels' greater than '1' in case '{self.__class__.__name__}' is not trainable"
            )

        self._init_p = p
        self.trainable = trainable
        self.channels = channels

        self.p = self.add_weight((channels,), "zeros") if trainable else p

    def call(self, x, index=None, axis=-2):
        p = self.p
        if self.channels != 1:
            assert ops.is_tensor(p)
            self.assert_two_dimensional_input(x, axis)
            p = ops.reshape(p, (-1, self.channels))
        if not isinstance(p, (int, float)) or p != 1.0:
            x = ops.clip(x, 0, 100) ** p
        out = self.reduce(x, index, axis, reduce_fn=_segment_mean)
        if not isinstance(p, (int, float)) or p != 1:
            out = ops.clip(out, 0, 100) ** (1.0 / p)
        return out


def get_unique(inputs):
    from keras import backend

    if backend.backend() == "tensorflow":
        import tensorflow as tf

        return tf.unique(inputs)
    elif backend.backend() == "torch":
        import torch

        return torch.unique(inputs, return_inverse=True)
    elif backend.backend() == "jax":
        import jax.numpy as jnp

        return jnp.unique(inputs, return_inverse=True)
    elif backend.backend() == "numpy":
        import numpy as np

        return np.unique(inputs, return_inverse=True)


def _segment_mean(data, segment_ids):
    data = ops.convert_to_tensor(data)
    segment_ids = ops.convert_to_tensor(segment_ids)
    unique_segment_ids, indices = get_unique(segment_ids)
    segment_sums = ops.segment_sum(data, indices, ops.shape(unique_segment_ids)[0])
    segment_counts = ops.segment_sum(
        ops.ones_like(data), indices, ops.shape(unique_segment_ids)[0]
    )
    segment_counts = ops.where(
        segment_counts > 0, segment_counts, ops.ones_like(segment_counts)
    )
    segment_means = segment_sums / segment_counts
    return segment_means
