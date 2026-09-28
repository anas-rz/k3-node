from typing import Optional
from keras import ops

from .base import Aggregation
from k3_node.ops.segment import segment_max, segment_sum


class AttentionalAggregation(Aggregation):
    r"""The soft attention aggregation layer from the `"Graph Matching Networks
    for Learning the Similarity of Graph Structured Objects"
    <https://arxiv.org/abs/1904.12787>`_ paper.

    Example:
        ```python
        import numpy as np
        import keras
        from k3_node.layers import AttentionalAggregation

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        index = np.repeat([0, 1], 5)  # aggregate nodes 0-4 into set 0 and nodes 5-9 into set 1

        aggr = AttentionalAggregation(gate_nn=keras.layers.Dense(1), nn=keras.layers.Dense(16))
        out = aggr(x, index=index, dim_size=2)  # attention-weighted sum per set
        print(tuple(out.shape))  # (2, 16)
        ```
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
        if dim_size is None:  # a tensor while tracing; don't test its truth value
            dim_size = int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0

        # Graph-wise softmax over groups
        max_val = segment_max(gate, index, num_segments=dim_size)
        max_exp = ops.take(max_val, index, axis=0)
        exp_gate = ops.exp(gate - max_exp)
        sum_exp = segment_sum(exp_gate, index, num_segments=dim_size)
        sum_exp_exp = ops.take(sum_exp, index, axis=0)
        alpha = exp_gate / ops.maximum(sum_exp_exp, 1e-12)

        return self.reduce(alpha * x, index, ptr, dim_size, dim, reduce="sum")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gate_nn={self.gate_nn}, nn={self.nn})"

