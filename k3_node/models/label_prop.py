from typing import Callable, Optional

import keras
from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm


class LabelPropagation(MessagePassing):
    r"""The label propagation operator, firstly introduced in the
    `"Learning from Labeled and Unlabeled Data with Label Propagation"
    <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`_ paper.

    .. math::
        \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},

    where unlabeled data is inferred by labeled data via propagation.
    This concrete implementation here is derived from the `"Combining Label
    Propagation And Simple Models Out-performs Graph Neural Networks"
    <https://arxiv.org/abs/2010.13993>`_ paper.

    Args:
        num_layers (int): The number of propagations.
        alpha (float): The :math:`\alpha` coefficient.
    """
    def __init__(self, num_layers: int, alpha: float, **kwargs):
        super().__init__(aggr='sum', **kwargs)
        self.num_layers = num_layers
        self.alpha = alpha

    def build(self, input_shape=None):
        self.built = True

    def call(
        self,
        y,
        edge_index,
        mask=None,
        edge_weight=None,
        post_step: Optional[Callable] = None,
    ):
        shape = ops.shape(y)
        if len(shape) == 1:
            num_classes = int(ops.max(y)) + 1
            y = ops.one_hot(y, num_classes)

        y = ops.cast(y, dtype="float32")
        out = y
        if mask is not None:
            mask_shape = ops.shape(mask)
            if len(mask_shape) == 1 and mask.dtype == "bool":
                mask_expanded = ops.expand_dims(mask, axis=-1)
                out = ops.where(mask_expanded, y, ops.zeros_like(y))
            else:
                out_zeros = ops.zeros_like(y)
                out = ops.scatter_update(out_zeros, ops.expand_dims(mask, -1), ops.take(y, mask, axis=0))

        if edge_weight is None:
            num_nodes = ops.shape(y)[0]
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight=None,
                num_nodes=num_nodes,
                add_self_loops=False,
                dtype=y.dtype,
            )

        res = (1.0 - self.alpha) * out
        for _ in range(self.num_layers):
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight)
            out = self.alpha * out + res
            if post_step is not None:
                out = post_step(out)
            else:
                out = ops.clip(out, 0.0, 1.0)

        return out

    def message(self, x_j, edge_weight=None):
        if edge_weight is None:
            return x_j
        return ops.expand_dims(edge_weight, axis=-1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_layers={self.num_layers}, '
                f'alpha={self.alpha})')

