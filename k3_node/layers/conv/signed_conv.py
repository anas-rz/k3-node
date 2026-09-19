from keras import ops
from keras.layers import Dense

from k3_node.layers.conv.message_passing import MessagePassing


class SignedConv(MessagePassing):
    r"""The signed graph convolutional operator from the `"Signed Graph
    Convolutional Network" <https://arxiv.org/abs/1808.06354>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        first_aggr: bool,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "mean")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_aggr = first_aggr
        self.use_bias = bias

        in_pos = in_channels if first_aggr else 2 * in_channels
        self.lin_pos_l = Dense(out_channels, use_bias=False)
        self.lin_pos_r = Dense(out_channels, use_bias=bias)
        self.lin_neg_l = Dense(out_channels, use_bias=False)
        self.lin_neg_r = Dense(out_channels, use_bias=bias)

    def build(self, input_shape=None):
        in_dim = self.in_channels if self.first_aggr else 2 * self.in_channels
        self.lin_pos_l.build((None, in_dim))
        self.lin_pos_r.build((None, self.in_channels))
        self.lin_neg_l.build((None, in_dim))
        self.lin_neg_r.build((None, self.in_channels))
        self.built = True

    def call(self, inputs, pos_edge_index=None, neg_edge_index=None, **kwargs):
        if pos_edge_index is None:
            if isinstance(inputs, (list, tuple)) and len(inputs) == 3:
                x, pos_edge_index, neg_edge_index = inputs
            else:
                raise ValueError("Expected (x, pos_edge_index, neg_edge_index)")
        else:
            x = inputs

        if not self.built:
            self.build()

        if isinstance(x, (list, tuple)):
            x_src, x_dst = x
        else:
            x_src = x_dst = x

        if self.first_aggr:
            out_pos = self.propagate(pos_edge_index, x=(x_src, x_dst))
            out_pos = self.lin_pos_l(out_pos) + self.lin_pos_r(x_dst)

            out_neg = self.propagate(neg_edge_index, x=(x_src, x_dst))
            out_neg = self.lin_neg_l(out_neg) + self.lin_neg_r(x_dst)

            return ops.concatenate([out_pos, out_neg], axis=-1)
        else:
            F_in = self.in_channels
            x_src_1, x_src_2 = x_src[..., :F_in], x_src[..., F_in:]
            x_dst_1, x_dst_2 = x_dst[..., :F_in], x_dst[..., F_in:]

            out_pos1 = self.propagate(pos_edge_index, x=(x_src_1, x_dst_1))
            out_pos2 = self.propagate(neg_edge_index, x=(x_src_2, x_dst_2))
            out_pos = ops.concatenate([out_pos1, out_pos2], axis=-1)
            out_pos = self.lin_pos_l(out_pos) + self.lin_pos_r(x_dst_1)

            out_neg1 = self.propagate(pos_edge_index, x=(x_src_2, x_dst_2))
            out_neg2 = self.propagate(neg_edge_index, x=(x_src_1, x_dst_1))
            out_neg = ops.concatenate([out_neg1, out_neg2], axis=-1)
            out_neg = self.lin_neg_l(out_neg) + self.lin_neg_r(x_dst_2)

            return ops.concatenate([out_pos, out_neg], axis=-1)

    def message(self, x_j):
        return x_j

