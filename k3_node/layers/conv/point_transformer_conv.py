from typing import Optional, Callable, Union, Tuple
from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.utils import remove_self_loops, add_self_loops, softmax


class PointTransformerConv(MessagePassing):
    r"""The Point Transformer layer from the `"Point Transformer"
    <https://arxiv.org/abs/2012.09164>`_ paper.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        pos_nn: Optional[Callable] = None,
        attn_nn: Optional[Callable] = None,
        add_self_loops: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        if pos_nn is None:
            self.pos_nn = Dense(out_channels, use_bias=True)
        else:
            self.pos_nn = pos_nn

        self.attn_nn = attn_nn
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.in_channels_l = in_channels
            self.in_channels_r = in_channels
        else:
            self.in_channels_l, self.in_channels_r = in_channels

        self.lin = Dense(out_channels, use_bias=False)
        self.lin_src = Dense(out_channels, use_bias=False)
        self.lin_dst = Dense(out_channels, use_bias=False)

    def build(self, input_shape=None):
        self.lin.build((None, self.in_channels_l))
        self.lin_src.build((None, self.in_channels_l))
        self.lin_dst.build((None, self.in_channels_r))
        if hasattr(self.pos_nn, "build") and not getattr(self.pos_nn, "built", False):
            self.pos_nn.build((None, 3))
        self.built = True

    def call(self, inputs, pos=None, edge_index=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)):
                if len(inputs) == 3:
                    x, pos, edge_index = inputs
                elif len(inputs) == 2:
                    x, edge_index = inputs
                else:
                    raise ValueError(f"Unexpected input length {len(inputs)}")
            else:
                raise ValueError("Expected (x, pos, edge_index)")
        else:
            x = inputs

        if not self.built:
            self.build()

        if isinstance(x, (list, tuple)):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        if isinstance(pos, (list, tuple)):
            pos_src, pos_dst = pos
        else:
            pos_src = pos_dst = pos

        num_nodes = ops.shape(pos_dst)[0]
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        alpha = (self.lin_src(x_src), self.lin_dst(x_dst))
        x_mapped = (self.lin(x_src), x_dst)

        out = self.propagate(
            edge_index,
            x=x_mapped,
            pos=(pos_src, pos_dst),
            alpha=alpha,
            size=(ops.shape(pos_src)[0], num_nodes),
        )

        return out

    def message(self, x_j, pos_i, pos_j, alpha_i, alpha_j, index=None, size_i=None):
        delta = pos_i - pos_j
        if self.pos_nn is not None:
            delta = self.pos_nn(delta)

        alpha = alpha_i - alpha_j + delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)

        alpha = softmax(alpha, index, num_nodes=size_i, dim=0)
        return (x_j + delta) * alpha
