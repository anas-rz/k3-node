from typing import Optional, Union, Callable
import keras
from keras import ops, activations
from keras.layers import Dropout

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm


class ARMAConv(MessagePassing):
    r"""The ARMA graph convolutional operator from the `"Graph Neural Networks
    with Convolutional ARMA Filters" <https://arxiv.org/abs/1901.01343>`_
    paper.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_stacks: int = 1,
        num_layers: int = 1,
        shared_weights: bool = False,
        act: Union[str, Callable, None] = "relu",
        dropout: float = 0.0,
        bias: bool = True,
        # Spektral compatibility arguments:
        channels: Optional[int] = None,
        order: Optional[int] = None,
        iterations: Optional[int] = None,
        share_weights: Optional[bool] = None,
        gcn_activation: Optional[str] = None,
        dropout_rate: Optional[float] = None,
        activation: Optional[str] = None,
        use_bias: Optional[bool] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        if out_channels is None:
            if channels is not None:
                out_channels = channels
                in_channels = -1
            else:
                out_channels = in_channels
                in_channels = -1

        if order is not None:
            num_stacks = order
        if iterations is not None:
            num_layers = iterations
        if share_weights is not None:
            shared_weights = share_weights
        if gcn_activation is not None:
            act = gcn_activation
        if dropout_rate is not None:
            dropout = dropout_rate
        if use_bias is not None:
            bias = use_bias

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_stacks = num_stacks
        self.num_layers = num_layers
        self.shared_weights = shared_weights
        self.act = activations.get(act) if act is not None else None
        self.dropout_rate = dropout
        self.use_bias = bias

        K, T, F_in, F_out = num_stacks, num_layers, in_channels, out_channels
        T_w = 1 if self.shared_weights else T

        self.weight = self.add_weight(
            shape=(max(1, T_w - 1), K, F_out, F_out),
            initializer="glorot_uniform",
            name="weight",
        )
        if in_channels is not None and in_channels != -1:
            self.init_weight = self.add_weight(
                shape=(K, F_in, F_out),
                initializer="glorot_uniform",
                name="init_weight",
            )
            self.root_weight = self.add_weight(
                shape=(T_w, K, F_in, F_out),
                initializer="glorot_uniform",
                name="root_weight",
            )
        else:
            self.init_weight = None
            self.root_weight = None

        if bias:
            self.bias = self.add_weight(
                shape=(T_w, K, 1, F_out),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None

        self.dropout = Dropout(dropout)

    def build(self, input_shape=None):
        if input_shape is not None:
            if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and isinstance(input_shape[0], (list, tuple)):
                dim = input_shape[0][-1]
            elif isinstance(input_shape, (list, tuple)) and len(input_shape) > 0 and input_shape[0] is not None and not isinstance(input_shape[0], (int, type(None))):
                dim = getattr(input_shape[0], "shape", [None, None])[-1]
            else:
                dim = input_shape[-1]
            if (self.in_channels is None or self.in_channels == -1) and dim is not None:
                self.in_channels = dim
        if self.in_channels is not None and self.in_channels != -1:
            K, T, F_in, F_out = self.num_stacks, self.num_layers, self.in_channels, self.out_channels
            T_w = 1 if self.shared_weights else T
            if self.init_weight is None:
                self.init_weight = self.add_weight(
                    shape=(K, F_in, F_out),
                    initializer="glorot_uniform",
                    name="init_weight",
                )
            if self.root_weight is None:
                self.root_weight = self.add_weight(
                    shape=(T_w, K, F_in, F_out),
                    initializer="glorot_uniform",
                    name="root_weight",
                )
        self.built = True

    def call(self, inputs, edge_index=None, edge_weight=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 3:
                    x, edge_index, edge_weight = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        if self.in_channels is None or self.in_channels == -1:
            self.build((None, ops.shape(x)[-1]))

        # Legacy adj check
        is_legacy = False
        if hasattr(edge_index, "shape") and len(edge_index.shape) == 2:
            if edge_index.shape[0] != 2 and edge_index.shape[0] == edge_index.shape[1]:
                is_legacy = True
        elif not hasattr(edge_index, "shape"):
            is_legacy = True

        if is_legacy:
            if hasattr(edge_index, "indices"):
                edge_weight = edge_index.values
                edge_index = ops.transpose(edge_index.indices)
            else:
                adj = edge_index
                row, col = ops.where(adj > 0)
                edge_index = ops.stack([row, col], axis=0)
                edge_weight = ops.take(adj, row * ops.shape(adj)[1] + col)

        num_nodes = ops.shape(x)[0]
        edge_index, edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes=num_nodes,
            add_self_loops=False,
            dtype=x.dtype,
        )

        # PyG: x = x.unsqueeze(-3) -> (1, N, F_in)
        # out = x
        # for t in range(num_layers):
        #   if t == 0: out = out @ init_weight   (K, F_in, F_out) -> (K, N, F_out)
        #   else: out = out @ weight[t-1]       (K, F_out, F_out) -> (K, N, F_out)
        #   out = propagate(edge_index, x=out, edge_weight=edge_weight)
        #   root = dropout(x) @ root_weight[t]  (K, F_in, F_out) -> (K, N, F_out)
        #   out = out + root
        #   if bias: out = out + bias[t]
        #   if act: out = act(out)
        # return out.mean(dim=-3)
        K, T, F_in, F_out = self.num_stacks, self.num_layers, self.in_channels, self.out_channels

        # out shape: (N, K, F_out)
        out = None
        for t in range(self.num_layers):
            w_idx = 0 if self.shared_weights else t
            if t == 0:
                # x: (N, F_in), init_weight: (K, F_in, F_out)
                # out: (K, N, F_out)
                out = ops.einsum("nf,kfo->kno", x, self.init_weight)
            else:
                w = self.weight[0 if self.shared_weights else t - 1]
                out = ops.einsum("kno,kof->knf", out, w)

            # Transpose to (N, K, F_out) so node_dim=0
            out_n = ops.transpose(out, (1, 0, 2))
            out_prop = self.propagate(edge_index, x=out_n, edge_weight=edge_weight, size=(num_nodes, num_nodes))
            out = ops.transpose(out_prop, (1, 0, 2))  # (K, N, F_out)

            root_x = self.dropout(x)
            root = ops.einsum("nf,kfo->kno", root_x, self.root_weight[w_idx])
            out = out + root

            if self.bias is not None:
                # bias[w_idx]: (K, 1, F_out)
                out = out + self.bias[w_idx]

            if self.act is not None:
                out = self.act(out)

        # out: (K, N, F_out) -> mean over K -> (N, F_out)
        return ops.mean(out, axis=0)

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else ops.expand_dims(ops.expand_dims(edge_weight, -1), -1) * x_j
