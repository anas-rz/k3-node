from typing import Optional
from keras import ops
import numpy as np

from .base import Aggregation


class SortAggregation(Aggregation):
    r"""The pooling operator from the `"An End-to-End Deep Learning
    Architecture for Graph Classification"
    <https://www.cse.wustl.edu/~muhan/papers/AAAI_2018_DGCNN.pdf>`_ paper,
    where node features are sorted in descending order based on their last
    feature channel. The first :math:`k` nodes form the output of the layer.
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
        fill_value = float(ops.min(x)) - 1.0
        batch_x, mask = self.to_dense_batch(
            x, index=index, ptr=ptr, dim_size=dim_size, dim=dim,
            fill_value=fill_value, max_num_elements=max_num_elements,
        )

        batch_x_np = ops.convert_to_numpy(batch_x)
        B, N, D = batch_x_np.shape

        # Sort along last feature channel descending
        scores = batch_x_np[:, :, -1]  # [B, N]
        perm = np.argsort(-scores, axis=-1)  # [B, N]

        # Gather sorted nodes for each graph in batch
        sorted_x = np.take_along_axis(batch_x_np, perm[:, :, None], axis=1)

        if N >= self.k:
            out_x = sorted_x[:, :self.k]
        else:
            pad = np.full((B, self.k - N, D), fill_value, dtype=batch_x_np.dtype)
            out_x = np.concatenate([sorted_x, pad], axis=1)

        out_x[out_x == fill_value] = 0.0
        out_x = out_x.reshape(B, self.k * D)

        return ops.convert_to_tensor(out_x, dtype=x.dtype)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"

