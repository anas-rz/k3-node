from typing import Union

from .base import Aggregation


def aggregation_resolver(
    query: Union[str, Aggregation],
    *args,
    **kwargs,
) -> Aggregation:
    r"""Resolves an aggregation string or instance to an `Aggregation` object.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import aggregation_resolver

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        index = np.repeat([0, 1], 5)  # aggregate nodes 0-4 into set 0 and nodes 5-9 into set 1

        aggr = aggregation_resolver("mean")  # build an Aggregation from its name
        print(type(aggr).__name__, tuple(aggr(x, index=index, dim_size=2).shape))  # MeanAggregation (2, 8)
        ```
    """
    if isinstance(query, Aggregation):
        return query

    if not isinstance(query, str):
        raise ValueError(f"Expected string or Aggregation instance, got {type(query)}")

    query_norm = query.lower().strip()

    from .basic import (
        MaxAggregation,
        MeanAggregation,
        MinAggregation,
        MulAggregation,
        PowerMeanAggregation,
        SoftmaxAggregation,
        StdAggregation,
        SumAggregation,
        VarAggregation,
    )
    from .quantile import MedianAggregation, QuantileAggregation
    from .variance_preserving import VariancePreservingAggregation

    AGGR_DICT = {
        "sum": SumAggregation,
        "add": SumAggregation,
        "mean": MeanAggregation,
        "max": MaxAggregation,
        "min": MinAggregation,
        "mul": MulAggregation,
        "var": VarAggregation,
        "std": StdAggregation,
        "softmax": SoftmaxAggregation,
        "powermean": PowerMeanAggregation,
        "median": MedianAggregation,
        "quantile": QuantileAggregation,
        "variance_preserving": VariancePreservingAggregation,
        "vpa": VariancePreservingAggregation,
    }

    if query_norm in AGGR_DICT:
        return AGGR_DICT[query_norm](*args, **kwargs)

    raise ValueError(f"Could not resolve aggregation '{query}'")

