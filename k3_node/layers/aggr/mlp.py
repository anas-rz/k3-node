from typing import Optional
from keras import layers, ops

from .base import Aggregation


class MLPAggregation(Aggregation):
    r"""Performs MLP aggregation in which the elements to aggregate are
    flattened into a single vectorial representation, and are then processed by
    a Multi-Layer Perceptron (MLP).

    Example:
        ```python
        import numpy as np
        from k3_node.layers import MLPAggregation

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        index = np.repeat([0, 1], 5)  # aggregate nodes 0-4 into set 0 and nodes 5-9 into set 1

        aggr = MLPAggregation(in_channels=8, out_channels=16, max_num_elements=5)
        out = aggr(x, index=index, dim_size=2)
        print(tuple(out.shape))  # (2, 16)
        ```
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

