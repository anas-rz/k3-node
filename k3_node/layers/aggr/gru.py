from typing import Optional
from keras import layers

from .base import Aggregation


class GRUAggregation(Aggregation):
    r"""Performs GRU aggregation in which the elements to aggregate are
    interpreted as a sequence, as described in the `"Graph Neural Networks
    with Adaptive Readouts" <https://arxiv.org/abs/2211.04952>`_ paper.
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gru = layers.GRU(out_channels, return_sequences=False)

    def build(self, input_shape=None):
        self.gru.build((None, None, self.in_channels))
        super().build(input_shape)

    def reset_parameters(self):
        self.gru.reset_parameters()

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
        return self.gru(x_dense, training=training)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"

