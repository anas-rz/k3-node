from typing import List, Union
from .base import Aggregation


class FusedAggregation(Aggregation):
    r"""Helper class to fuse computation of multiple aggregations together.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import FusedAggregation

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        index = np.repeat([0, 1], 5)  # aggregate nodes 0-4 into set 0 and nodes 5-9 into set 1

        aggr = FusedAggregation(aggrs=["sum", "mean", "max"])
        outs = aggr(x, index=index, dim_size=2)  # one result per aggregation, computed together
        print(len(outs), tuple(outs[0].shape))  # 3 (2, 8)
        ```
    """

    def __init__(self, aggrs: List[Union[Aggregation, str]], **kwargs):
        super().__init__(**kwargs)
        from .resolver import aggregation_resolver

        self.aggrs = [aggregation_resolver(aggr) for aggr in aggrs]

    def reset_parameters(self):
        for aggr in self.aggrs:
            if hasattr(aggr, "reset_parameters"):
                aggr.reset_parameters()

    def call(
        self,
        x,
        index=None,
        ptr=None,
        dim_size=None,
        dim=-2,
        **kwargs,
    ):
        return [aggr(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim, **kwargs) for aggr in self.aggrs]

