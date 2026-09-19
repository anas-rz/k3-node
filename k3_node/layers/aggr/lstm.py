from typing import Optional
from keras import layers

from .base import Aggregation


class LSTMAggregation(Aggregation):
    r"""Performs LSTM-style aggregation in which the elements to aggregate are
    interpreted as a sequence, as described in the `"Inductive Representation
    Learning on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper.
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lstm = layers.LSTM(out_channels, return_sequences=False)

    def build(self, input_shape=None):
        self.lstm.build((None, None, self.in_channels))
        super().build(input_shape)

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
        training: bool = False,
        **kwargs,
    ):
        x_dense, _ = self.to_dense_batch(
            x, index=index, ptr=ptr, dim_size=dim_size, dim=dim,
            max_num_elements=max_num_elements,
        )
        return self.lstm(x_dense, training=training)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"

