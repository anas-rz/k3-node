from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm


class PDNConv(MessagePassing):
    r"""The pathfinder discovery network convolutional operator from the
    `"Pathfinder Discovery Networks for Neural Message Passing"
    <https://arxiv.org/abs/2010.12878>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        hidden_channels: int,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.hidden_channels = hidden_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.use_bias = bias

        self.mlp_1 = Dense(hidden_channels, activation="relu")
        self.mlp_2 = Dense(1, activation="sigmoid")
        self.lin = Dense(out_channels, use_bias=False)

        if bias:
            self.bias = self.add_weight(
                shape=(out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape=None):
        self.mlp_1.build((None, self.edge_dim))
        self.mlp_2.build((None, self.hidden_channels))
        self.lin.build((None, self.in_channels))
        self.built = True

    def call(self, inputs, edge_index=None, edge_attr=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 3:
                    x, edge_index, edge_attr = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        if not self.built:
            self.build()

        if edge_attr is not None:
            edge_attr = self.mlp_1(edge_attr)
            edge_attr = ops.squeeze(self.mlp_2(edge_attr), -1)

        num_nodes = ops.shape(x)[0]
        if self.normalize:
            edge_index, edge_attr = gcn_norm(
                edge_index,
                edge_attr,
                num_nodes=num_nodes,
                add_self_loops=self.add_self_loops,
                dtype=x.dtype,
            )

        x = self.lin(x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_attr)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else ops.expand_dims(edge_weight, -1) * x_j

