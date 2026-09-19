from typing import Optional
from keras import ops

from .base import Aggregation


class AttentionalAggregation(Aggregation):
    r"""The soft attention aggregation layer from the `"Graph Matching Networks
    for Learning the Similarity of Graph Structured Objects"
    <https://arxiv.org/abs/1904.12787>`_ paper.
    """

    def __init__(
        self,
        gate_nn,
        nn: Optional[any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gate_nn = gate_nn
        self.nn = nn

    def reset_parameters(self):
        if hasattr(self.gate_nn, "reset_parameters"):
            self.gate_nn.reset_parameters()
        if self.nn is not None and hasattr(self.nn, "reset_parameters"):
            self.nn.reset_parameters()

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        gate = self.gate_nn(x)
        if self.nn is not None:
            x = self.nn(x)

        if ptr is not None and index is None:
            from .base import ptr2index
            index = ptr2index(ptr)

        index = ops.cast(index, dtype="int32")
        dim_size = dim_size or (int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0)

        # Graph-wise softmax over groups
        max_val = ops.segment_max(gate, index, num_segments=dim_size)
        max_exp = ops.take(max_val, index, axis=0)
        exp_gate = ops.exp(gate - max_exp)
        sum_exp = ops.segment_sum(exp_gate, index, num_segments=dim_size)
        sum_exp_exp = ops.take(sum_exp, index, axis=0)
        alpha = exp_gate / ops.maximum(sum_exp_exp, 1e-12)

        return self.reduce(alpha * x, index, ptr, dim_size, dim, reduce="sum")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gate_nn={self.gate_nn}, nn={self.nn})"

