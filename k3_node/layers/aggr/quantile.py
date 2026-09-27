from typing import List, Optional, Union
from keras import ops
import numpy as np

from .base import Aggregation


class QuantileAggregation(Aggregation):
    r"""An aggregation operator that returns the feature-wise :math:`q`-th
    quantile of a set :math:`\mathcal{X}`.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import QuantileAggregation

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        index = np.repeat([0, 1], 5)  # aggregate nodes 0-4 into set 0 and nodes 5-9 into set 1

        aggr = QuantileAggregation(q=0.75)
        out = aggr(x, index=index, dim_size=2)
        print(tuple(out.shape))  # (2, 8)
        ```
    """
    interpolations = {"linear", "lower", "higher", "nearest", "midpoint"}

    def __init__(
        self,
        q: Union[float, List[float]],
        interpolation: str = "linear",
        fill_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        qs = [q] if not isinstance(q, (list, tuple)) else list(q)
        if len(qs) == 0:
            raise ValueError("Provide at least one quantile value for `q`.")
        if not all(0.0 <= quantile <= 1.0 for quantile in qs):
            raise ValueError("`q` must be in the range [0, 1].")
        if interpolation not in self.interpolations:
            raise ValueError(f"Invalid interpolation method got ('{interpolation}')")

        self.q = qs
        self.interpolation = interpolation
        self.fill_value = fill_value

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        self.assert_index_present(index)
        from k3_node.ops.segment import segment_sum

        # Sort every set's values in a dense [sets, max_size, features] tensor; the padding sorts
        # last (+inf) and is then zeroed. Static shapes and differentiable, like PyG's version.
        dense, _ = self.to_dense_batch(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim,
                                       fill_value=float("inf"))
        dense = ops.sort(dense, axis=1)
        dense = ops.where(ops.isinf(dense), ops.zeros_like(dense), dense)

        index_i = ops.cast(index, "int32")
        count = segment_sum(ops.ones_like(index_i), index_i, num_segments=ops.shape(dense)[0])
        count_f = ops.cast(count, dense.dtype)
        last = ops.maximum(count - 1, 0)

        def gather(position):  # the value at `position` of every set, per feature
            position = ops.minimum(ops.maximum(ops.cast(position, "int32"), 0), last)
            position = ops.broadcast_to(ops.reshape(position, (-1, 1, 1)),
                                        (ops.shape(dense)[0], 1, ops.shape(dense)[2]))
            return ops.take_along_axis(dense, position, axis=1)[:, 0]

        outs = []
        for q_val in self.q:
            q_point = q_val * (count_f - 1.0)
            if self.interpolation == "lower":
                quantile = gather(ops.floor(q_point))
            elif self.interpolation == "higher":
                quantile = gather(ops.ceil(q_point))
            elif self.interpolation == "nearest":
                quantile = gather(ops.round(q_point))
            else:
                low, high = gather(ops.floor(q_point)), gather(ops.ceil(q_point))
                if self.interpolation == "linear":
                    frac = ops.expand_dims(q_point - ops.floor(q_point), -1)
                    quantile = low + (high - low) * frac
                else:  # midpoint
                    quantile = 0.5 * low + 0.5 * high
            empty = ops.expand_dims(count == 0, -1)
            outs.append(ops.where(empty, ops.cast(self.fill_value, quantile.dtype), quantile))
        return outs[0] if len(outs) == 1 else ops.concatenate(outs, axis=-1)

    def __repr__(self) -> str:
        q_str = self.q[0] if len(self.q) == 1 else self.q
        return f"{self.__class__.__name__}(q={q_str})"


class MedianAggregation(QuantileAggregation):
    r"""An aggregation operator that returns the feature-wise median of a set.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import MedianAggregation

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        index = np.repeat([0, 1], 5)  # aggregate nodes 0-4 into set 0 and nodes 5-9 into set 1

        aggr = MedianAggregation()
        out = aggr(x, index=index, dim_size=2)
        print(tuple(out.shape))  # (2, 8)
        ```
    """

    def __init__(self, fill_value: float = 0.0, **kwargs):
        super().__init__(0.5, "lower", fill_value=fill_value, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

