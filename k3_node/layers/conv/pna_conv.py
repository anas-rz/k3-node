from typing import Optional, Union, List, Callable
import keras
from keras import ops, activations
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.aggr import DegreeScalerAggregation


class PNAConv(MessagePassing):
    r"""The Principal Neighbourhood Aggregation graph convolutional operator
    from the `"Principal Neighbourhood Aggregation for Graph Nets"
    <https://arxiv.org/abs/2004.05718>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: List[str],
        scalers: List[str],
        deg,
        edge_dim: Optional[int] = None,
        towers: int = 1,
        pre_layers: int = 1,
        post_layers: int = 1,
        divide_input: bool = False,
        act: Union[str, Callable, None] = "relu",
        train_norm: bool = False,
        **kwargs,
    ):
        aggr = DegreeScalerAggregation(aggregators, scalers, deg, train_norm)
        super().__init__(aggr=aggr, node_dim=0, **kwargs)

        if divide_input:
            assert in_channels % towers == 0
        assert out_channels % towers == 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregators = aggregators
        self.scalers = scalers
        self.edge_dim = edge_dim
        self.towers = towers
        self.divide_input = divide_input
        self.act = activations.get(act) if act is not None else None

        self.F_in = in_channels // towers if divide_input else in_channels
        self.F_out = out_channels // towers

        if edge_dim is not None:
            self.edge_encoder = Dense(self.F_in, use_bias=False)
        else:
            self.edge_encoder = None

        self.pre_nns = []
        self.post_nns = []
        for _ in range(towers):
            pre_dim = (3 if edge_dim else 2) * self.F_in
            pre_layers_list = [Dense(self.F_in, use_bias=True)]
            for _ in range(pre_layers - 1):
                pre_layers_list.append(Dense(self.F_in, activation=self.act, use_bias=True))
            self.pre_nns.append(pre_layers_list)

            post_in_dim = (len(aggregators) * len(scalers) + 1) * self.F_in
            post_layers_list = [Dense(self.F_out, use_bias=True)]
            for _ in range(post_layers - 1):
                post_layers_list.append(Dense(self.F_out, activation=self.act, use_bias=True))
            self.post_nns.append(post_layers_list)

        self.lin = Dense(out_channels, use_bias=True)

    def build(self, input_shape=None):
        if self.edge_encoder is not None:
            self.edge_encoder.build((None, self.edge_dim))
        for pre_list in self.pre_nns:
            dim = (3 if self.edge_dim else 2) * self.F_in
            for l in pre_list:
                l.build((None, dim))
                dim = self.F_in
        for post_list in self.post_nns:
            dim = (len(self.aggregators) * len(self.scalers) + 1) * self.F_in
            for l in post_list:
                l.build((None, dim))
                dim = self.F_out
        self.lin.build((None, self.out_channels))
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

        num_nodes = ops.shape(x)[0]
        if self.divide_input:
            x_towers = ops.reshape(x, (-1, self.towers, self.F_in))
        else:
            x_towers = ops.repeat(ops.expand_dims(x, 1), self.towers, axis=1)

        out = self.propagate(
            edge_index,
            x=x_towers,
            edge_attr=edge_attr,
            size=(num_nodes, num_nodes),
        )

        out = ops.concatenate([x_towers, out], axis=-1)

        outs = []
        for i, post_list in enumerate(self.post_nns):
            h_i = out[:, i]
            for l in post_list:
                h_i = l(h_i)
            outs.append(h_i)

        out = ops.concatenate(outs, axis=-1)
        return self.lin(out)

    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None and self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = ops.repeat(ops.expand_dims(edge_attr, 1), self.towers, axis=1)
            h = ops.concatenate([x_i, x_j, edge_attr], axis=-1)
        else:
            h = ops.concatenate([x_i, x_j], axis=-1)

        hs = []
        for i, pre_list in enumerate(self.pre_nns):
            h_i = h[:, i]
            for l in pre_list:
                h_i = l(h_i)
            hs.append(h_i)

        return ops.stack(hs, axis=1)

