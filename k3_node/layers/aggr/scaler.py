from typing import Any, Dict, List, Optional, Union
from keras import initializers, ops
import numpy as np

from .base import Aggregation
from k3_node.ops.segment import segment_sum


class DegreeScalerAggregation(Aggregation):
    r"""Combines one or more aggregators and transforms its output with one or
    more scalers as introduced in the `"Principal Neighbourhood Aggregation for
    Graph Nets" <https://arxiv.org/abs/2004.05718>`_ paper.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import DegreeScalerAggregation

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        index = np.repeat([0, 1], 5)  # aggregate nodes 0-4 into set 0 and nodes 5-9 into set 1

        deg = np.array([0, 3, 5, 2])  # in-degree histogram of the training graphs
        aggr = DegreeScalerAggregation(aggr=["mean", "max"], scaler=["identity", "amplification"], deg=deg)
        out = aggr(x, index=index, dim_size=2)
        print(tuple(out.shape))  # (2, 32): 2 aggregators x 2 scalers x 8 features
        ```
    """

    def __init__(
        self,
        aggr: Union[str, List[str], Aggregation],
        scaler: Union[str, List[str]],
        deg,
        train_norm: bool = False,
        aggr_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        from .resolver import aggregation_resolver
        from .multi import MultiAggregation

        if isinstance(aggr, (str, Aggregation)):
            self.aggr = aggregation_resolver(aggr, **(aggr_kwargs or {}))
        elif isinstance(aggr, (tuple, list)):
            self.aggr = MultiAggregation(aggr, aggr_kwargs)
        else:
            raise ValueError(
                f"Only strings, list, tuples and instances of "
                f"`Aggregation` are valid aggregation schemes (got '{type(aggr)}')"
            )

        self.scaler = [scaler] if isinstance(scaler, str) else list(scaler)

        deg_np = ops.convert_to_numpy(deg).astype(np.float32)
        N = float(np.sum(deg_np))
        bin_degree = np.arange(len(deg_np), dtype=np.float32)

        self.init_avg_deg_lin = float(np.sum(bin_degree * deg_np)) / max(N, 1.0)
        self.init_avg_deg_log = float(np.sum(np.log(bin_degree + 1.0) * deg_np)) / max(N, 1.0)
        self.train_norm = train_norm

        if train_norm:
            self.avg_deg_lin = self.add_weight(
                shape=(1,),
                initializer=initializers.Constant(self.init_avg_deg_lin),
                trainable=True,
                name="avg_deg_lin",
            )
            self.avg_deg_log = self.add_weight(
                shape=(1,),
                initializer=initializers.Constant(self.init_avg_deg_log),
                trainable=True,
                name="avg_deg_log",
            )
        else:
            self.avg_deg_lin = self.init_avg_deg_lin
            self.avg_deg_log = self.init_avg_deg_log

    def reset_parameters(self):
        if hasattr(self.aggr, "reset_parameters"):
            self.aggr.reset_parameters()
        if self.train_norm:
            self.avg_deg_lin.assign(ops.full((1,), self.init_avg_deg_lin, dtype=self.avg_deg_lin.dtype))
            self.avg_deg_log.assign(ops.full((1,), self.init_avg_deg_log, dtype=self.avg_deg_log.dtype))

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

        out = self.aggr(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim)

        index = ops.cast(index, dtype="int32")
        dim_size = dim_size or (int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0)

        # Compute degree per index
        ones = ops.ones((ops.shape(index)[0],), dtype=out.dtype)
        deg = segment_sum(ones, index, num_segments=dim_size)
        deg = ops.reshape(deg, (dim_size,) + (1,) * (len(ops.shape(out)) - 1))

        avg_deg_log = self.avg_deg_log
        avg_deg_lin = self.avg_deg_lin

        outs = []
        for scaler in self.scaler:
            if scaler == "identity":
                out_scaler = out
            elif scaler == "amplification":
                out_scaler = out * (ops.log(deg + 1.0) / avg_deg_log)
            elif scaler == "attenuation":
                out_scaler = out * (avg_deg_log / ops.log(ops.maximum(deg, 1.0) + 1.0))
            elif scaler == "linear":
                out_scaler = out * (deg / avg_deg_lin)
            elif scaler == "inverse_linear":
                out_scaler = out * (avg_deg_lin / ops.maximum(deg, 1.0))
            else:
                raise ValueError(f"Unknown scaler '{scaler}'")
            outs.append(out_scaler)

        return ops.concatenate(outs, axis=-1) if len(outs) > 1 else outs[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(aggr={self.aggr}, scaler={self.scaler})"

