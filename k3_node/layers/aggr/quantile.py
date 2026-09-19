from typing import List, Optional, Union
from keras import ops
import numpy as np

from .base import Aggregation


class QuantileAggregation(Aggregation):
    r"""An aggregation operator that returns the feature-wise :math:`q`-th
    quantile of a set :math:`\mathcal{X}`.
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
        x_np = ops.convert_to_numpy(x)
        idx_np = ops.convert_to_numpy(index).astype(np.int64)

        B = int(np.max(idx_np)) + 1 if len(idx_np) > 0 else 0
        if dim_size is not None:
            B = max(B, dim_size)

        outs = []
        for q_val in self.q:
            q_out = np.full((B, *x_np.shape[1:]), self.fill_value, dtype=x_np.dtype)
            for b in range(B):
                mask = idx_np == b
                if not np.any(mask):
                    continue
                group_x = x_np[mask]
                # Sort along axis 0
                sorted_x = np.sort(group_x, axis=0)
                n = sorted_x.shape[0]
                q_pos = q_val * (n - 1)
                i_low = int(np.floor(q_pos))
                i_high = int(np.ceil(q_pos))

                if self.interpolation == "lower":
                    q_out[b] = sorted_x[i_low]
                elif self.interpolation == "higher":
                    q_out[b] = sorted_x[i_high]
                elif self.interpolation == "nearest":
                    idx_round = int(round(q_pos))
                    q_out[b] = sorted_x[idx_round]
                elif self.interpolation == "midpoint":
                    q_out[b] = 0.5 * sorted_x[i_low] + 0.5 * sorted_x[i_high]
                else:  # linear
                    frac = q_pos - i_low
                    q_out[b] = sorted_x[i_low] + frac * (sorted_x[i_high] - sorted_x[i_low])
            outs.append(q_out)

        if len(outs) == 1:
            return ops.convert_to_tensor(outs[0], dtype=x.dtype)
        else:
            cat_out = np.concatenate(outs, axis=-1)
            return ops.convert_to_tensor(cat_out, dtype=x.dtype)

    def __repr__(self) -> str:
        q_str = self.q[0] if len(self.q) == 1 else self.q
        return f"{self.__class__.__name__}(q={q_str})"


class MedianAggregation(QuantileAggregation):
    r"""An aggregation operator that returns the feature-wise median of a set."""

    def __init__(self, fill_value: float = 0.0, **kwargs):
        super().__init__(0.5, "lower", fill_value=fill_value, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

