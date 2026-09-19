from typing import Union

from .base import Aggregation


def aggregation_resolver(
    query: Union[str, Aggregation],
    *args,
    **kwargs,
) -> Aggregation:
    r"""Resolves an aggregation string or instance to an `Aggregation` object."""
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

