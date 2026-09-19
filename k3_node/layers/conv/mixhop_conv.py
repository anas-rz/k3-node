from typing import Optional, List
import keras
from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm


class MixHopConv(MessagePassing):
    r"""The MixHop graph convolutional operator from the
    `"Higher-Order Graph Convolutional Networks via MixHop"
    <https://arxiv.org/abs/1905.00067>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        powers: Optional[List[int]] = None,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        if powers is None:
            powers = [0, 1, 2]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.powers = powers
        self.add_self_loops = add_self_loops
        self.use_bias = bias

        self.lins = [Dense(out_channels, use_bias=False) for _ in range(max(powers) + 1)]

        if bias:
            self.bias = self.add_weight(
                shape=(len(powers) * out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape=None):
        for lin in self.lins:
            lin.build((None, self.in_channels))
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

        if not self.built:
            self.build()

        num_nodes = ops.shape(x)[0]
        edge_index, edge_weight = gcn_norm(
            edge_index,
            edge_weight,
            num_nodes=num_nodes,
            add_self_loops=self.add_self_loops,
            dtype=x.dtype,
        )

        outs = [self.lins[0](x)]
        curr_x = x
        for lin in self.lins[1:]:
            curr_x = self.propagate(edge_index, x=curr_x, edge_weight=edge_weight)
            outs.append(lin(curr_x))

        out = ops.concatenate([outs[p] for p in self.powers], axis=-1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, edge_weight=None):
        return x_j if edge_weight is None else ops.expand_dims(edge_weight, -1) * x_j

