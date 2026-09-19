import copy
from typing import Any, Dict, List, Optional, Union
from keras import layers, ops

from .base import Aggregation


class MultiAggregation(Aggregation):
    r"""Performs aggregations with one or more aggregators and combines
    aggregated results, as described in the `"Principal Neighbourhood
    Aggregation for Graph Nets" <https://arxiv.org/abs/2004.05718>`_ and
    `"Adaptive Filters and Aggregator Fusion for Efficient Graph Convolutions"
    <https://arxiv.org/abs/2104.01481>`_ papers.
    """

    def __init__(
        self,
        aggrs: List[Union[Aggregation, str]],
        aggrs_kwargs: Optional[List[Dict[str, Any]]] = None,
        mode: Optional[str] = "cat",
        mode_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(aggrs, (list, tuple)):
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should be a list or tuple.")

        if len(aggrs) == 0:
            raise ValueError(f"'aggrs' of '{self.__class__.__name__}' should not be empty.")

        if aggrs_kwargs is None:
            aggrs_kwargs = [{}] * len(aggrs)
        elif len(aggrs) != len(aggrs_kwargs):
            raise ValueError(
                f"'aggrs_kwargs' with invalid length passed to '{self.__class__.__name__}' "
                f"(got '{len(aggrs_kwargs)}', expected '{len(aggrs)}')."
            )

        from .resolver import aggregation_resolver
        self.aggrs = [
            aggregation_resolver(aggr, **aggr_kw)
            for aggr, aggr_kw in zip(aggrs, aggrs_kwargs)
        ]

        self.mode = mode
        mode_kwargs = copy.copy(mode_kwargs) or {}
        self.in_channels = mode_kwargs.pop("in_channels", None)
        self.out_channels = mode_kwargs.pop("out_channels", None)

        if mode in ["proj", "attn"]:
            if len(aggrs) == 1:
                raise ValueError("Multiple aggregations are required for 'proj' or 'attn' combine mode.")
            if self.in_channels is None or self.out_channels is None:
                raise ValueError(f"Combine mode '{mode}' must have `in_channels` and `out_channels` specified.")

            if isinstance(self.in_channels, int):
                self.in_channels = [self.in_channels] * len(aggrs)

            if mode == "proj":
                self.lin = layers.Dense(self.out_channels, **mode_kwargs)
            elif mode == "attn":
                from ..dense import HeteroDictLinear
                channels = {str(k): v for k, v in enumerate(self.in_channels)}
                self.lin_heads = HeteroDictLinear(channels, self.out_channels)
                num_heads = mode_kwargs.pop("num_heads", 1)
                self.multihead_attn = layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=max(self.out_channels // num_heads, 1),
                    **mode_kwargs,
                )

    def reset_parameters(self):
        for aggr in self.aggrs:
            if hasattr(aggr, "reset_parameters"):
                aggr.reset_parameters()
        if hasattr(self, "lin") and hasattr(self.lin, "reset_parameters"):
            self.lin.reset_parameters()
        if hasattr(self, "lin_heads") and hasattr(self.lin_heads, "reset_parameters"):
            self.lin_heads.reset_parameters()

    def get_out_channels(self, in_channels: int) -> int:
        if self.out_channels is not None:
            return self.out_channels
        if self.mode == "cat":
            return in_channels * len(self.aggrs)
        return in_channels

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        outs = [aggr(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim, **kwargs) for aggr in self.aggrs]
        return self.combine(outs)

    def combine(self, inputs: List[any]):
        if len(inputs) == 1:
            return inputs[0]

        if self.mode == "cat":
            return ops.concatenate(inputs, axis=-1)

        if hasattr(self, "lin"):
            return self.lin(ops.concatenate(inputs, axis=-1))

        if hasattr(self, "multihead_attn"):
            x_dict = {str(k): v for k, v in enumerate(inputs)}
            x_dict = self.lin_heads(x_dict)
            xs = [x_dict[str(key)] for key in range(len(inputs))]
            # xs: [num_aggrs, B, D] -> transpose to [B, num_aggrs, D]
            x_stack = ops.transpose(ops.stack(xs, axis=0), (1, 0, 2))
            attn_out = self.multihead_attn(x_stack, x_stack, x_stack)
            return ops.mean(attn_out, axis=1)

        stacked = ops.stack(inputs, axis=0)  # [num_aggrs, B, D]
        if self.mode == "sum":
            return ops.sum(stacked, axis=0)
        elif self.mode == "mean":
            return ops.mean(stacked, axis=0)
        elif self.mode == "max":
            return ops.max(stacked, axis=0)
        elif self.mode == "min":
            return ops.min(stacked, axis=0)
        elif self.mode == "logsumexp":
            return ops.logsumexp(stacked, axis=0)
        elif self.mode == "std":
            return ops.std(stacked, axis=0)
        elif self.mode == "var":
            return ops.var(stacked, axis=0)

        raise ValueError(f"Combine mode '{self.mode}' is not supported.")

    def __repr__(self) -> str:
        aggrs = ",\n".join([f"  {aggr}" for aggr in self.aggrs])
        return f"{self.__class__.__name__}([\n{aggrs}\n], mode={self.mode})"

