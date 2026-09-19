from typing import List, Union
from .base import Aggregation


class FusedAggregation(Aggregation):
    r"""Helper class to fuse computation of multiple aggregations together."""

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

