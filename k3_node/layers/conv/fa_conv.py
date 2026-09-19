from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm


class FAConv(MessagePassing):
    r"""The Frequency Adaptive Graph Convolution operator from the
    `"Beyond Low-Frequency Information in Graph Convolutional Networks"
    <https://arxiv.org/abs/2101.00797>`_ paper.
    """
    def __init__(
        self,
        channels: int,
        eps: float = 0.1,
        dropout: float = 0.0,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.channels = channels
        self.eps = eps
        self.dropout_rate = dropout
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.att_l = Dense(1, use_bias=False)
        self.att_r = Dense(1, use_bias=False)

    def build(self, input_shape=None):
        self.att_l.build((None, self.channels))
        self.att_r.build((None, self.channels))
        self.built = True

    def call(self, inputs, x_0=None, edge_index=None, edge_weight=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 4:
                    x, x_0, edge_index, edge_weight = inputs
                elif len(inputs) == 3:
                    x, x_0, edge_index = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                    x_0 = x
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected inputs with edge_index")
        else:
            x = inputs
            if x_0 is None:
                x_0 = x

        if not self.built:
            self.build()

        num_nodes = ops.shape(x)[0]
        if self.normalize:
            edge_index, edge_weight = gcn_norm(
                edge_index,
                edge_weight,
                num_nodes=num_nodes,
                add_self_loops=self.add_self_loops,
                dtype=x.dtype,
            )

        alpha_l = self.att_l(x)
        alpha_r = self.att_r(x)

        out = self.propagate(
            edge_index,
            x=x,
            alpha=(alpha_l, alpha_r),
            edge_weight=edge_weight,
        )

        if self.eps != 0.0:
            out = out + self.eps * x_0

        return out

    def message(self, x_j, alpha_j, alpha_i, edge_weight=None):
        alpha = ops.tanh(alpha_j + alpha_i)
        if edge_weight is not None:
            alpha = alpha * ops.expand_dims(edge_weight, -1)
        return x_j * alpha

