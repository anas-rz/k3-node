from typing import Callable, Optional, Tuple, Union
from keras import layers, ops
import numpy as np

from .connect.filter_edges import FilterEdges
from .select.topk import SelectTopK


class LEConv(layers.Layer):
    r"""The local extremum graph neural network operator."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = layers.Dense(out_channels, use_bias=bias, name="lin1")
        self.lin2 = layers.Dense(out_channels, use_bias=False, name="lin2")
        self.lin3 = layers.Dense(out_channels, use_bias=bias, name="lin3")

    def reset_parameters(self):
        pass

    def call(self, x, edge_index, edge_weight: Optional[any] = None):
        a = self.lin1(x)
        b = self.lin2(x)
        row = ops.cast(edge_index[0], dtype="int32")
        col = ops.cast(edge_index[1], dtype="int32")
        num_nodes = ops.shape(x)[0]

        a_j = ops.take(a, row, axis=0)
        b_i = ops.take(b, col, axis=0)
        msg = a_j - b_i
        if edge_weight is not None:
            msg = msg * ops.reshape(edge_weight, (-1, 1))

        out = ops.segment_sum(msg, col, num_segments=num_nodes)
        return out + self.lin3(x)


class ASAPooling(layers.Layer):
    r"""The Adaptive Structure Aware Pooling operator from the
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical
    Graph Representations" <https://arxiv.org/abs/1911.07979>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        GNN: Optional[Callable] = None,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        add_self_loops: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.negative_slope = negative_slope
        self.dropout_rate = dropout
        self.drop = layers.Dropout(dropout) if dropout > 0.0 else None
        self.add_self_loops = add_self_loops

        self.lin = layers.Dense(in_channels, use_bias=True, name="lin")
        self.att = layers.Dense(1, use_bias=True, name="att")
        self.gnn_score = LEConv(in_channels, 1, name="gnn_score")

        if GNN is not None:
            self.gnn_intra_cluster = GNN(in_channels, in_channels, **kwargs)
        else:
            self.gnn_intra_cluster = None

        self.select = SelectTopK(1, ratio)

    def reset_parameters(self):
        self.select.reset_parameters()

    def call(
        self,
        x,
        edge_index,
        edge_weight: Optional[any] = None,
        batch: Optional[any] = None,
        training: bool = False,
    ) -> Tuple[any, any, Optional[any], any, any]:
        r"""Forward pass."""
        N = ops.shape(x)[0]
        if batch is None:
            batch = ops.zeros((N,), dtype="int32")
        else:
            batch = ops.cast(batch, dtype="int32")

        if len(ops.shape(x)) == 1:
            x = ops.expand_dims(x, axis=-1)

        # Add self-loops to edge_index and edge_weight
        loop_idx = ops.arange(N, dtype=edge_index.dtype)
        loop_edge = ops.stack([loop_idx, loop_idx], axis=0)
        edge_index = ops.concatenate([edge_index, loop_edge], axis=1)

        num_edges = ops.shape(edge_index)[1]
        if edge_weight is None:
            edge_weight = ops.ones((num_edges,), dtype=x.dtype)
        else:
            edge_weight = ops.concatenate([edge_weight, ops.ones((N,), dtype=x.dtype)], axis=0)

        x_pool = x
        if self.gnn_intra_cluster is not None:
            x_pool = self.gnn_intra_cluster(x=x, edge_index=edge_index, edge_weight=edge_weight)

        row = ops.cast(edge_index[0], dtype="int32")
        col = ops.cast(edge_index[1], dtype="int32")

        x_pool_j = ops.take(x_pool, row, axis=0)
        x_q = ops.segment_max(x_pool_j, col, num_segments=N)
        x_q = ops.take(self.lin(x_q), col, axis=0)

        score = ops.reshape(self.att(ops.concatenate([x_q, x_pool_j], axis=-1)), (-1,))
        score = ops.leaky_relu(score, negative_slope=self.negative_slope)

        # Softmax over col
        score_max = ops.segment_max(score, col, num_segments=N)
        score_max_exp = ops.take(score_max, col, axis=0)
        exp_score = ops.exp(score - score_max_exp)
        sum_exp = ops.segment_sum(exp_score, col, num_segments=N)
        sum_exp_exp = ops.take(sum_exp, col, axis=0)
        score = exp_score / (sum_exp_exp + 1e-12)

        if self.drop is not None:
            score = self.drop(score, training=training)

        v_j = ops.take(x, row, axis=0) * ops.reshape(score, (-1, 1))
        x_new = ops.segment_sum(v_j, col, num_segments=N)

        fitness = ops.reshape(ops.sigmoid(self.gnn_score(x_new, edge_index)), (-1,))
        select_out = self.select(fitness, batch)
        perm = select_out.node_index

        x_out = ops.take(x_new, perm, axis=0) * ops.reshape(ops.take(fitness, perm, axis=0), (-1, 1))

        connect = FilterEdges()
        connect_out = connect(select_out, edge_index, edge_weight, batch)

        edge_index_out = connect_out.edge_index
        edge_weight_out = connect_out.edge_attr
        batch_out = connect_out.batch

        return x_out, edge_index_out, edge_weight_out, batch_out, perm

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, ratio={self.ratio})"

