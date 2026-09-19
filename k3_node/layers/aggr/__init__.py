r"""Aggregation operators for graph neural networks."""

from .base import Aggregation, ptr2index, to_dense_batch
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
from .attention import AttentionalAggregation
from .set2set import Set2Set
from .scaler import DegreeScalerAggregation
from .sort import SortAggregation
from .multi import MultiAggregation
from .deep_sets import DeepSetsAggregation
from .mlp import MLPAggregation
from .lstm import LSTMAggregation
from .gru import GRUAggregation
from .set_transformer import SetTransformerAggregation
from .gmt import GraphMultisetTransformer
from .variance_preserving import VariancePreservingAggregation
from .patch_transformer import PatchTransformerAggregation
from .lcm import LCMAggregation
from .equilibrium import EquilibriumAggregation
from .fused import FusedAggregation
from .resolver import aggregation_resolver

__all__ = [
    "Aggregation",
    "ptr2index",
    "to_dense_batch",
    "SumAggregation",
    "MeanAggregation",
    "MaxAggregation",
    "MinAggregation",
    "MulAggregation",
    "VarAggregation",
    "StdAggregation",
    "SoftmaxAggregation",
    "PowerMeanAggregation",
    "QuantileAggregation",
    "MedianAggregation",
    "AttentionalAggregation",
    "Set2Set",
    "DegreeScalerAggregation",
    "SortAggregation",
    "MultiAggregation",
    "DeepSetsAggregation",
    "MLPAggregation",
    "LSTMAggregation",
    "GRUAggregation",
    "SetTransformerAggregation",
    "GraphMultisetTransformer",
    "VariancePreservingAggregation",
    "PatchTransformerAggregation",
    "LCMAggregation",
    "EquilibriumAggregation",
    "FusedAggregation",
    "aggregation_resolver",
]

classes = __all__
