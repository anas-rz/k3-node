from typing import Optional
from keras import ops

from .base import Aggregation


class VariancePreservingAggregation(Aggregation):
    r"""Performs the Variance Preserving Aggregation (VPA) from the `"GNN-VPA:
    A Variance-Preserving Aggregation Strategy for Graph Neural Networks"
    <https://arxiv.org/abs/2403.04747>`_ paper.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import VariancePreservingAggregation

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        index = np.repeat([0, 1], 5)  # aggregate nodes 0-4 into set 0 and nodes 5-9 into set 1

        aggr = VariancePreservingAggregation()
        out = aggr(x, index=index, dim_size=2)
        print(tuple(out.shape))  # (2, 8)
        ```
    """

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        out = self.reduce(x, index, ptr, dim_size, dim, reduce="sum")

        if ptr is not None and index is None:
            from .base import ptr2index
            index = ptr2index(ptr)

        index = ops.cast(index, dtype="int32")
        dim_size = dim_size or (int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0)

        ones = ops.ones((ops.shape(index)[0], 1), dtype=out.dtype)
        count = ops.segment_sum(ones, index, num_segments=dim_size)
        count = ops.maximum(ops.sqrt(count), 1.0)

        return out / count

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

