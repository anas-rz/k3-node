from keras import layers, ops
from .linear import Linear


class DenseGCNConv(layers.Layer):
    r"""Applies the dense convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\tilde{D}}^{-1/2} \mathbf{\tilde{A}}
        \mathbf{\tilde{D}}^{-1/2} \mathbf{X} \mathbf{\Theta}

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\tilde{A}} = \mathbf{A} + 2 \mathbf{I}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.use_bias = bias

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot', name="lin")

        if bias:
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
        if self.bias is not None:
            self.bias.assign(ops.zeros(self.bias.shape, dtype=self.bias.dtype))

    def build(self, input_shape):
        if isinstance(input_shape, (tuple, list)):
            x_shape = input_shape[0]
        else:
            x_shape = input_shape
        self.lin.build(x_shape)
        super().build(input_shape)

    def call(self, x, adj, mask=None, add_loop: bool = True):
        is_2d_x = (len(ops.shape(x)) == 2)
        if is_2d_x:
            x = ops.expand_dims(x, axis=0)
        if len(ops.shape(adj)) == 2:
            adj = ops.expand_dims(adj, axis=0)

        N = ops.shape(adj)[1]

        if add_loop:
            diag_val = 2.0 if self.improved else 1.0
            eye = ops.expand_dims(ops.eye(N, dtype=adj.dtype), axis=0)
            adj = adj * (1.0 - eye) + diag_val * eye

        out = self.lin(x)
        deg = ops.maximum(ops.sum(adj, axis=-1), 1.0)
        deg_inv_sqrt = ops.power(deg, -0.5)

        adj_norm = ops.expand_dims(deg_inv_sqrt, axis=-1) * adj * ops.expand_dims(deg_inv_sqrt, axis=-2)
        out = ops.matmul(adj_norm, out)

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
        return (*x_shape[:-1], self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels})')
