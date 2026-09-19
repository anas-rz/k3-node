from typing import Optional
from keras import layers, ops

from .base import Aggregation


class Set2Set(Aggregation):
    r"""The Set2Set aggregation operator based on iterative content-based
    attention, as described in the `"Order Matters: Sequence to sequence for
    Sets" <https://arxiv.org/abs/1511.06391>`_ paper.
    """

    def __init__(self, in_channels: int, processing_steps: int, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.lstm_cell = layers.LSTMCell(in_channels)

    def build(self, input_shape=None):
        self.lstm_cell.build((None, self.out_channels))
        super().build(input_shape)

    def reset_parameters(self):
        self.lstm_cell.reset_parameters()

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        if ptr is not None and index is None:
            from .base import ptr2index
            index = ptr2index(ptr)

        self.assert_index_present(index)
        self.assert_two_dimensional_input(x, dim)

        index = ops.cast(index, dtype="int32")
        dim_size = dim_size or (int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0)

        # Initial hidden states: [dim_size, in_channels]
        h = [
            ops.zeros((dim_size, self.in_channels), dtype=x.dtype),
            ops.zeros((dim_size, self.in_channels), dtype=x.dtype),
        ]
        q_star = ops.zeros((dim_size, self.out_channels), dtype=x.dtype)

        for _ in range(self.processing_steps):
            q, h = self.lstm_cell(q_star, h)
            q_taken = ops.take(q, index, axis=0)
            e = ops.sum(x * q_taken, axis=-1, keepdims=True)

            max_e = ops.segment_max(e, index, num_segments=dim_size)
            max_e_exp = ops.take(max_e, index, axis=0)
            exp_e = ops.exp(e - max_e_exp)
            sum_exp = ops.segment_sum(exp_e, index, num_segments=dim_size)
            sum_exp_exp = ops.take(sum_exp, index, axis=0)
            a = exp_e / ops.maximum(sum_exp_exp, 1e-12)

            r = ops.segment_sum(a * x, index, num_segments=dim_size)
            q_star = ops.concatenate([q, r], axis=-1)

        return q_star

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"

