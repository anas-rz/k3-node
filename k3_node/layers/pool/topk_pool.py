from typing import Callable, Optional, Tuple, Union
from keras import layers, ops

from .connect.filter_edges import FilterEdges
from .select.topk import SelectTopK


class TopKPooling(layers.Layer):
    r""":math:`\mathrm{top}_k` pooling operator from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_, `"Towards Sparse
    Hierarchical Graph Classifiers" <https://arxiv.org/abs/1811.01287>`_
    and `"Understanding Attention and Generalization in Graph Neural
    Networks" <https://arxiv.org/abs/1905.02850>`_ papers.
    """
    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
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

        self.select = SelectTopK(in_channels, ratio, min_score, nonlinearity)
        self.connect = FilterEdges()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.select.reset_parameters()

    def build(self, input_shape=None):
        if not self.select.built:
            self.select.build(input_shape)
        if hasattr(self.connect, "built") and not self.connect.built:
            self.connect.build(None)
        self.built = True

    def call(
        self,
        x,
        edge_index,
        edge_attr: Optional[any] = None,
        batch: Optional[any] = None,
        attn: Optional[any] = None,
    ) -> Tuple[any, any, Optional[any], Optional[any], any, any]:
        r"""Forward pass."""
        if batch is None:
            batch = ops.zeros((ops.shape(x)[0],), dtype="int32")

        attn_input = x if attn is None else attn
        select_out = self.select(attn_input, batch)

        perm = select_out.node_index
        score = select_out.weight

        x_pooled = ops.take(x, perm, axis=0) * ops.expand_dims(score, axis=-1)
        if self.multiplier != 1.0:
            x_pooled = x_pooled * self.multiplier

        connect_out = self.connect.call(select_out, edge_index, edge_attr, batch)

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
        return (
            f"{self.__class__.__name__}({self.in_channels}, {ratio}, "
            f"multiplier={self.multiplier})"
        )

