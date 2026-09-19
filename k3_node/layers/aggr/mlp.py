from typing import Optional
from keras import layers, ops

from .base import Aggregation


class MLPAggregation(Aggregation):
    r"""Performs MLP aggregation in which the elements to aggregate are
    flattened into a single vectorial representation, and are then processed by
    a Multi-Layer Perceptron (MLP).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_num_elements: int,
        mlp: Optional[any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_num_elements = max_num_elements

        if mlp is None:
            self.mlp = layers.Dense(out_channels)
        else:
            self.mlp = mlp

    def build(self, input_shape=None):
        if hasattr(self.mlp, "build"):
            self.mlp.build((None, self.in_channels * self.max_num_elements))
        super().build(input_shape)

    def reset_parameters(self):
        if hasattr(self.mlp, "reset_parameters"):
            self.mlp.reset_parameters()

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        x_dense, _ = self.to_dense_batch(
            x, index=index, ptr=ptr, dim_size=dim_size, dim=dim,
            max_num_elements=self.max_num_elements,
        )
        B = ops.shape(x_dense)[0]
        flattened = ops.reshape(x_dense, (B, self.max_num_elements * self.in_channels))
        return self.mlp(flattened)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, "
            f"max_num_elements={self.max_num_elements})"
        )

