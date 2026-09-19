from typing import Optional
from keras import layers, ops
from .linear import Linear


class DenseGraphConv(layers.Layer):
    r"""See :class:`torch_geometric.nn.conv.GraphConv`."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'add',
        bias: bool = True,
        **kwargs
    ):
        if aggr not in ('add', 'mean', 'max'):
            raise ValueError(f"aggr must be one of 'add', 'mean', 'max' (got '{aggr}')")
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        self.use_bias = bias

        self.lin_rel = Linear(in_channels, out_channels, bias=bias, name="lin_rel")
        self.lin_root = Linear(in_channels, out_channels, bias=False, name="lin_root")

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape
        self.lin_rel.build(x_shape)
        self.lin_root.build(x_shape)
        super().build(input_shape)

    def call(self, x, adj, mask=None):
        is_2d_x = (len(ops.shape(x)) == 2)
        if is_2d_x:
            x = ops.expand_dims(x, axis=0)
        if len(ops.shape(adj)) == 2:
            adj = ops.expand_dims(adj, axis=0)

        N = ops.shape(x)[1]

        if self.aggr == 'add':
            out = ops.matmul(adj, x)
        elif self.aggr == 'mean':
            deg = ops.maximum(ops.sum(adj, axis=-1, keepdims=True), 1.0)
            out = ops.matmul(adj, x) / deg
        elif self.aggr == 'max':
            x_src = ops.expand_dims(x, axis=1)  # [B, 1, N, C]
            adj_expanded = ops.expand_dims(adj, axis=-1)  # [B, N, N, 1]
            masked = ops.where(adj_expanded != 0, x_src, -1e9)
            out = ops.max(masked, axis=2)  # [B, N, C]
            out = ops.where(out <= -1e8, 0.0, out)

        out = self.lin_rel(out) + self.lin_root(x)

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
        return (*x_shape[:-1], self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
