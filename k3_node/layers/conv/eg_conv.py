from typing import Optional, List
from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import gcn_norm, scatter


class EGConv(MessagePassing):
    r"""The Efficient Graph Convolution from the `"Adaptive Filters and
    Aggregator Fusion for Efficient Graph Convolutions"
    <https://arxiv.org/abs/2104.01481>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregators: Optional[List[str]] = None,
        num_heads: int = 8,
        num_bases: int = 4,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        if out_channels % num_heads != 0:
            raise ValueError(
                f"'out_channels' ({out_channels}) must be divisible by num_heads ({num_heads})"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.aggregators = aggregators or ["symnorm"]
        self.use_bias = bias

        self.bases_lin = Dense(
            (out_channels // num_heads) * num_bases, use_bias=False
        )
        self.comb_lin = Dense(
            num_heads * num_bases * len(self.aggregators), use_bias=True
        )

        if bias:
            self.bias = self.add_weight(
                shape=(out_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape=None):
        self.bases_lin.build((None, self.in_channels))
        self.comb_lin.build((None, self.in_channels))
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

        num_nodes = ops.shape(x)[0]
        symnorm_weight = None
        if "symnorm" in self.aggregators:
            edge_index, symnorm_weight = gcn_norm(
                edge_index,
                edge_weight=None,
                num_nodes=num_nodes,
                add_self_loops=self.add_self_loops,
                dtype=x.dtype,
            )

        bases = self.bases_lin(x)
        weightings = self.comb_lin(x)

        aggregated = self.propagate(
            edge_index,
            x=bases,
            symnorm_weight=symnorm_weight,
            size=(num_nodes, num_nodes),
        )

        weightings = ops.reshape(
            weightings,
            (-1, self.num_heads, self.num_bases * len(self.aggregators)),
        )
        aggregated = ops.reshape(
            aggregated,
            (
                -1,
                len(self.aggregators) * self.num_bases,
                self.out_channels // self.num_heads,
            ),
        )

        out = ops.matmul(weightings, aggregated)
        out = ops.reshape(out, (-1, self.out_channels))

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, edge_index=None, index=None, dim_size=None, symnorm_weight=None, **kwargs):
        if index is None and edge_index is not None:
            index = edge_index[1]

        outs = []
        for aggr in self.aggregators:
            if aggr == "symnorm":
                inp = inputs if symnorm_weight is None else inputs * ops.expand_dims(symnorm_weight, -1)
                out = scatter(inp, index, dim=0, dim_size=dim_size, reduce="sum")
            elif aggr in ("var", "std"):
                mean = scatter(inputs, index, dim=0, dim_size=dim_size, reduce="mean")
                mean_sq = scatter(inputs * inputs, index, dim=0, dim_size=dim_size, reduce="mean")
                out = mean_sq - mean * mean
                if aggr == "std":
                    out = ops.sqrt(ops.maximum(out, 1e-5))
            else:
                out = scatter(inputs, index, dim=0, dim_size=dim_size, reduce=aggr)
            outs.append(out)

        if len(outs) > 1:
            return ops.stack(outs, axis=1)
        return outs[0]

