from typing import Callable, Optional, Tuple, Union
from keras import initializers, layers, ops

from .connect.filter_edges import FilterEdges
from .select.topk import SelectTopK


class PANPooling(layers.Layer):
    r"""The path integral based pooling operator from the
    `"Path Integral Based Convolution and Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/2006.16811>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = "tanh",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.p = self.add_weight(
            shape=(in_channels,),
            initializer=initializers.Constant(1.0),
            trainable=True,
            name="p",
        )
        self.beta = self.add_weight(
            shape=(2,),
            initializer=initializers.Constant(0.5),
            trainable=True,
            name="beta",
        )

        self.select = SelectTopK(1, ratio, min_score, nonlinearity)
        self.connect = FilterEdges()

    def reset_parameters(self):
        self.p.assign(ops.ones(self.p.shape, dtype=self.p.dtype))
        self.beta.assign(ops.full(self.beta.shape, 0.5, dtype=self.beta.dtype))
        self.select.reset_parameters()

    def call(
        self,
        x,
        M,
        batch: Optional[any] = None,
    ) -> Tuple[any, any, any, Optional[any], any, any]:
        r"""Forward pass.

        Args:
            x: Node feature matrix.
            M: MET matrix, either a tuple/list (edge_index, edge_weight) or an object
               with `.coo()` method (like PyG SparseTensor).
            batch: Batch vector.
        """
        num_nodes = ops.shape(x)[0]
        if batch is None:
            batch = ops.zeros((num_nodes,), dtype="int32")

        if hasattr(M, "coo"):
            row, col, edge_weight = M.coo()
        elif isinstance(M, (tuple, list)):
            if len(M) == 2:
                edge_index, edge_weight = M
                row, col = edge_index[0], edge_index[1]
            else:
                row, col, edge_weight = M[0], M[1], M[2]
        else:
            nz = ops.where(M != 0)
            row, col = nz[0], nz[1]
            edge_weight = ops.take(ops.reshape(M, (-1,)), row * ops.shape(M)[1] + col, axis=0)

        col = ops.cast(col, dtype="int32")
        row = ops.cast(row, dtype="int32")

        score1 = ops.sum(x * self.p, axis=-1)
        score2 = ops.segment_sum(edge_weight, col, num_segments=num_nodes)
        score = self.beta[0] * score1 + self.beta[1] * score2

        select_out = self.select(score, batch)

        perm = select_out.node_index
        score_val = select_out.weight

        x_pooled = ops.take(x, perm, axis=0) * ops.expand_dims(score_val, axis=-1)
        if self.multiplier != 1.0:
            x_pooled = x_pooled * self.multiplier

        edge_index = ops.stack([col, row], axis=0)
        connect_out = self.connect(select_out, edge_index, edge_weight, batch)

        return (
            x_pooled,
            connect_out.edge_index,
            connect_out.edge_attr,
            connect_out.batch,
            perm,
            score_val,
        )

    def __repr__(self) -> str:
        if self.min_score is None:
            ratio = f"ratio={self.ratio}"
        else:
            ratio = f"min_score={self.min_score}"
        return (
            f"{self.__class__.__name__}({self.in_channels}, {ratio}, "
            f"multiplier={self.multiplier})"
        )
