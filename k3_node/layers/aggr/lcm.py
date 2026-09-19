from math import ceil, log2
from typing import Optional
from keras import layers, ops

from .base import Aggregation


class LCMAggregation(Aggregation):
    r"""The Learnable Commutative Monoid aggregation from the
    `"Learnable Commutative Monoids for Graph Neural Networks"
    <https://arxiv.org/abs/2212.08541>`_ paper.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        project: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if in_channels != out_channels and not project:
            raise ValueError(
                f"Inputs of '{self.__class__.__name__}' must be projected if `in_channels != out_channels`"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.project = project

        self.lin = layers.Dense(out_channels) if project else None
        self.gru_cell = layers.GRUCell(out_channels)

    def reset_parameters(self):
        if self.lin is not None:
            self.lin.reset_parameters()
        self.gru_cell.reset_parameters()

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
        **kwargs,
    ):
        if self.lin is not None:
            x = ops.relu(self.lin(x))

        x_dense, _ = self.to_dense_batch(
            x, index=index, ptr=ptr, dim_size=dim_size, dim=dim,
            max_num_elements=max_num_elements,
        )

        # Transpose to [num_neighbors, num_nodes, num_features]
        x_dense = ops.transpose(x_dense, (1, 0, 2))
        num_neighbors = ops.shape(x_dense)[0]
        num_nodes = ops.shape(x_dense)[1]
        num_features = ops.shape(x_dense)[2]

        if num_neighbors == 0:
            return ops.zeros((num_nodes, self.out_channels), dtype=x.dtype)

        depth = ceil(log2(max(num_neighbors, 1)))
        for _ in range(depth):
            curr_len = ops.shape(x_dense)[0]
            if curr_len <= 1:
                break
            half_size = ceil(curr_len / 2)

            if curr_len % 2 == 1:
                x_pair = x_dense[:-1]
                remainder = x_dense[-1:]
            else:
                x_pair = x_dense
                remainder = None

            # x_pair: [2 * half, num_nodes, num_features]
            pair_count = ops.shape(x_pair)[0] // 2
            x_pair = ops.reshape(x_pair, (pair_count, 2, num_nodes, num_features))
            left = x_pair[:, 0]  # [pair_count, num_nodes, num_features]
            right = x_pair[:, 1]  # [pair_count, num_nodes, num_features]

            left_flat = ops.reshape(left, (-1, num_features))
            right_flat = ops.reshape(right, (-1, num_features))

            # GRUCell: inputs=left, state=[right]
            out1, _ = self.gru_cell(left_flat, [right_flat])
            out2, _ = self.gru_cell(right_flat, [left_flat])
            out = 0.5 * (out1 + out2)
            out = ops.reshape(out, (pair_count, num_nodes, num_features))

            if remainder is not None:
                out = ops.concatenate([out, remainder], axis=0)

            x_dense = out

        return ops.squeeze(x_dense, axis=0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, project={self.project})"

