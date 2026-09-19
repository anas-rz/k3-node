from typing import Optional
from keras import initializers, layers, ops
from .linear import Linear


class DenseGATConv(layers.Layer):
    r"""See :class:`torch_geometric.nn.conv.GATConv`."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.use_bias = bias

        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot', name="lin")

        self.att_src = self.add_weight(
            shape=(1, 1, heads, out_channels),
            initializer="glorot_uniform",
            trainable=True,
            name="att_src",
        )
        self.att_dst = self.add_weight(
            shape=(1, 1, heads, out_channels),
            initializer="glorot_uniform",
            trainable=True,
            name="att_dst",
        )

        if bias and concat:
            self.bias = self.add_weight(
                shape=(heads * out_channels,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
        elif bias and not concat:
            self.bias = self.add_weight(
                shape=(out_channels,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot = initializers.GlorotUniform()
        self.att_src.assign(glorot(self.att_src.shape, dtype=self.att_src.dtype))
        self.att_dst.assign(glorot(self.att_dst.shape, dtype=self.att_dst.dtype))
        if self.bias is not None:
            self.bias.assign(ops.zeros(self.bias.shape, dtype=self.bias.dtype))

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape
        self.lin.build(x_shape)
        super().build(input_shape)

    def call(self, x, adj, mask: Optional[any] = None, add_loop: bool = True, training=None):
        is_2d_x = (len(ops.shape(x)) == 2)
        if is_2d_x:
            x = ops.expand_dims(x, axis=0)
        if len(ops.shape(adj)) == 2:
            adj = ops.expand_dims(adj, axis=0)

        H, C = self.heads, self.out_channels
        B = ops.shape(x)[0]
        N = ops.shape(x)[1]

        if add_loop:
            eye = ops.expand_dims(ops.eye(N, dtype=adj.dtype), axis=0)
            adj = adj * (1.0 - eye) + eye

        x_proj = ops.reshape(self.lin(x), (B, N, H, C))

        alpha_src = ops.sum(x_proj * self.att_src, axis=-1)  # [B, N, H]
        alpha_dst = ops.sum(x_proj * self.att_dst, axis=-1)  # [B, N, H]

        alpha = ops.expand_dims(alpha_src, axis=1) + ops.expand_dims(alpha_dst, axis=2)  # [B, N, N, H]
        alpha = ops.leaky_relu(alpha, negative_slope=self.negative_slope)
        alpha = ops.where(ops.expand_dims(adj, axis=-1) != 0, alpha, -1e9)
        alpha = ops.softmax(alpha, axis=2)

        # Transpose to [B, H, N, N] and [B, H, N, C]
        alpha_perm = ops.transpose(alpha, (0, 3, 1, 2))
        x_perm = ops.transpose(x_proj, (0, 2, 1, 3))
        out = ops.matmul(alpha_perm, x_perm)  # [B, H, N, C]
        out = ops.transpose(out, (0, 2, 1, 3))  # [B, N, H, C]

        if self.concat:
            out = ops.reshape(out, (B, N, H * C))
        else:
            out = ops.mean(out, axis=2)

        if self.bias is not None:
            out = out + self.bias

        if mask is not None:
            out = out * ops.cast(ops.reshape(mask, (-1, N, 1)), x.dtype)

        if is_2d_x and ops.shape(out)[0] == 1:
            out = ops.squeeze(out, axis=0)

        return out

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, (tuple, list)):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape
        out_dim = self.heads * self.out_channels if self.concat else self.out_channels
        return (*x_shape[:-1], out_dim)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')
