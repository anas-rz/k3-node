from typing import Callable, Optional, Tuple, Union
from keras import layers, ops

from .connect.filter_edges import FilterEdges
from .select.topk import SelectTopK


class GraphConv(layers.Layer):
    r"""Basic GraphConv layer for projection scoring in pooling layers."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = "add",
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr

        self.lin_rel = layers.Dense(out_channels, use_bias=bias, name="lin_rel")
        self.lin_root = layers.Dense(out_channels, use_bias=False, name="lin_root")

    def reset_parameters(self):
        pass

    def call(self, x, edge_index, edge_weight: Optional[any] = None):
        row = ops.cast(edge_index[0], dtype="int32")
        col = ops.cast(edge_index[1], dtype="int32")
        num_nodes = ops.shape(x)[0]

        msg = ops.take(x, row, axis=0)
        if edge_weight is not None:
            msg = msg * ops.reshape(edge_weight, (-1, 1))

        if self.aggr == "add":
            aggr_out = ops.segment_sum(msg, col, num_segments=num_nodes)
        elif self.aggr == "mean":
            sum_val = ops.segment_sum(msg, col, num_segments=num_nodes)
            count = ops.segment_sum(ops.ones_like(msg), col, num_segments=num_nodes)
            aggr_out = sum_val / ops.maximum(count, 1.0)
        elif self.aggr == "max":
            aggr_out = ops.segment_max(msg, col, num_segments=num_nodes)
        else:
            aggr_out = ops.segment_sum(msg, col, num_segments=num_nodes)

        return self.lin_rel(aggr_out) + self.lin_root(x)


class SAGPooling(layers.Layer):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers.
    """
    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        GNN: Optional[any] = None,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = "tanh",
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        if GNN is None:
            self.gnn = GraphConv(in_channels, 1, **kwargs)
        elif callable(GNN):
            self.gnn = GNN(in_channels, 1, **kwargs)
        else:
            self.gnn = GNN

        self.select = SelectTopK(1, ratio, min_score, nonlinearity)
        self.connect = FilterEdges()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.gnn, "reset_parameters"):
            self.gnn.reset_parameters()
        self.select.reset_parameters()

    def call(
        self,
        x,
        edge_index,
        edge_attr: Optional[any] = None,
        batch: Optional[any] = None,
        attn: Optional[any] = None,
    ) -> Tuple[any, any, Optional[any], Optional[any], any, any]:
        r"""Forward pass."""
        num_nodes = ops.shape(x)[0]
        if batch is None:
            batch = ops.zeros((num_nodes,), dtype="int32")

        if attn is None:
            attn = x
        if len(ops.shape(attn)) == 1:
            attn = ops.expand_dims(attn, axis=-1)

        attn = self.gnn(attn, edge_index)

        select_out = self.select(attn, batch)

        perm = select_out.node_index
        score = select_out.weight

        x_pooled = ops.take(x, perm, axis=0) * ops.expand_dims(score, axis=-1)
        if self.multiplier != 1.0:
            x_pooled = x_pooled * self.multiplier

        connect_out = self.connect(select_out, edge_index, edge_attr, batch)

        return (
            x_pooled,
            connect_out.edge_index,
            connect_out.edge_attr,
            connect_out.batch,
            perm,
            score,
        )

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f"ratio={self.ratio}"
        else:
            ratio = f"min_score={self.min_score}"
        gnn_name = self.gnn.__class__.__name__
        return (
            f"{self.__class__.__name__}({gnn_name}, {self.in_channels}, "
            f"{ratio}, multiplier={self.multiplier})"
        )

