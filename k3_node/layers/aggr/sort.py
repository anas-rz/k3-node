from typing import Optional
from keras import ops
import numpy as np

from .base import Aggregation
from k3_node.ops.creation import repeat


class SortAggregation(Aggregation):
    r"""The pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are sorted in descending order based on their last
    feature channel. The first :math:`k` nodes form the output of the layer.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import SortAggregation

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        index = np.repeat([0, 1], 5)  # aggregate nodes 0-4 into set 0 and nodes 5-9 into set 1

        aggr = SortAggregation(k=2)
        out = aggr(x, index=index, dim_size=2)
        print(tuple(out.shape))  # (2, 16)
        ```
    """

    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)
        self.k = k

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
        fill_value = -1e9
        batch_x, mask = self.to_dense_batch(
            x, index=index, ptr=ptr, dim_size=dim_size, dim=dim,
            fill_value=fill_value, max_num_elements=max_num_elements,
        )

        B = ops.shape(batch_x)[0]
        N = ops.shape(batch_x)[1]
        D = ops.shape(batch_x)[2]

        scores = batch_x[:, :, -1]  # [B, N]
        k = self.k
        k_val = min(k, int(scores.shape[1])) if hasattr(scores, "shape") and isinstance(scores.shape[1], int) else k
        _, perm = ops.top_k(scores, k=k_val, sorted=True)  # [B, k]
        perm_expanded = repeat(ops.expand_dims(perm, -1), D, axis=-1)
        out_x = ops.take_along_axis(batch_x, perm_expanded, axis=1)
        out_x = ops.where(ops.equal(out_x, fill_value), ops.zeros_like(out_x), out_x)
        out_x = ops.reshape(out_x, (B, -1))

        return out_x

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"

