from typing import Optional
from keras import ops

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import remove_self_loops, add_self_loops, softmax


class AGNNConv(MessagePassing):
    r"""The graph attentional propagation layer from the
    `"Attention-based Graph Neural Network for Semi-Supervised Learning"
    <https://arxiv.org/abs/1803.03735>`_ paper.
    """
    def __init__(
        self,
        requires_grad: bool = True,
        add_self_loops: bool = True,
        trainable: Optional[bool] = None,
        aggregate: str = "add",
        activation=None,
        **kwargs,
    ):
        if trainable is not None:
            requires_grad = trainable
        kwargs.setdefault("aggr", aggregate)
        super().__init__(activation=activation, **kwargs)

        self.requires_grad = requires_grad
        self.add_self_loops = add_self_loops

        if requires_grad:
            self.beta = self.add_weight(
                shape=(1,),
                initializer="ones",
                name="beta",
            )
        else:
            self.beta = None

    def build(self, input_shape=None):
        self.built = True

    def call(self, inputs, edge_index=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                x, edge_index = inputs
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        if not self.built:
            self.build()

        # Check for legacy 2D matrix
        is_legacy = False
        if hasattr(edge_index, "shape") and len(edge_index.shape) == 2:
            if (
                edge_index.shape[0] is not None
                and edge_index.shape[1] is not None
                and edge_index.shape[0] != 2
                and edge_index.shape[0] == edge_index.shape[1]
            ):
                is_legacy = True
        elif not hasattr(edge_index, "shape"):
            is_legacy = True

        x_norm = x / (ops.norm(x, axis=-1, keepdims=True) + 1e-12)

        if is_legacy:
            out = self.propagate(x, edge_index, x_norm=x_norm)
        else:
            num_nodes = x.shape[0] if hasattr(x, "shape") and x.shape[0] is not None else ops.shape(x)[0]
            if self.add_self_loops:
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            out = self.propagate(edge_index, x=x, x_norm=x_norm, size=(num_nodes, num_nodes))

        if self.activation is not None:
            out = self.activation(out)

        return out

    def message(self, x=None, x_j=None, x_norm=None, x_norm_i=None, x_norm_j=None, index=None, size_i=None):
        beta = self.beta if self.beta is not None else 1.0

        # Legacy Spektral path
        if x_j is None and x is not None:
            x_j = self.get_sources(x)
            x_norm_i = self.get_targets(x_norm)
            x_norm_j = self.get_sources(x_norm)
            alpha = beta * ops.sum(x_norm_i * x_norm_j, axis=-1)
            alpha = softmax(alpha, self.index_targets, num_nodes=self.n_nodes, dim=0)
            return ops.expand_dims(alpha, -1) * x_j

        # PyG path
        alpha = beta * ops.sum(x_norm_i * x_norm_j, axis=-1)
        alpha = softmax(alpha, index, num_nodes=size_i, dim=0)
        return x_j * ops.expand_dims(alpha, -1)
