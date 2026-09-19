from typing import List, Optional, Tuple, Union
from keras import layers, ops

from .linear import Linear


class MLP(layers.Layer):
    r"""A simple Multi-Layer Perceptron (MLP) model."""
    def __init__(
        self,
        channel_list: List[int],
        act: Optional[Union[str, any]] = None,
        norm: Optional[Union[str, any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.channel_list = channel_list
        self.lins = [
            Linear(in_c, out_c)
            for in_c, out_c in zip(channel_list[:-1], channel_list[1:])
        ]
        self.act = layers.Activation(act) if act is not None else None

    @property
    def in_channels(self) -> int:
        r"""Size of each input sample."""
        return self.channel_list[0]

    @property
    def out_channels(self) -> int:
        r"""Size of each output sample."""
        return self.channel_list[-1]

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for lin in self.lins:
            if hasattr(lin, "reset_parameters"):
                lin.reset_parameters()

    def call(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if self.act is not None and i < len(self.lins) - 1:
                x = self.act(x)
        return x


class DMoNPooling(layers.Layer):
    r"""The spectral modularity pooling operator from the `"Graph Clustering
    with Graph Neural Networks" <https://arxiv.org/abs/2006.16904>`_ paper.

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    Args:
        channels (int or List[int]): Size of each input sample. If given as a
            list, will construct an MLP based on the given feature sizes.
        k (int): The number of clusters.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)
    """
    def __init__(
        self,
        channels: Union[int, List[int]],
        k: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(channels, int):
            channels = [channels]

        self.channels = channels
        self.k = k
        self.dropout = dropout
        self.mlp = MLP(channels + [k], act=None, norm=None)
        self.drop = layers.Dropout(dropout) if dropout > 0.0 else None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mlp.reset_parameters()

    def call(
        self,
        x,
        adj,
        mask: Optional[any] = None,
        training: bool = False,
    ) -> Tuple[any, any, any, any, any, any]:
        r"""Forward pass.

        Args:
            x: Node feature tensor [B, N, F] or [N, F].
            adj: Adjacency tensor [B, N, N] or [N, N].
            mask: Mask tensor [B, N] indicating valid nodes. (default: None)
            training: Whether layer is in training mode.
        """
        if len(ops.shape(x)) == 2:
            x = ops.expand_dims(x, axis=0)
        if len(ops.shape(adj)) == 2:
            adj = ops.expand_dims(adj, axis=0)

        s = self.mlp(x)
        if self.drop is not None:
            s = self.drop(s, training=training)
        s = ops.softmax(s, axis=-1)

        batch_size = ops.shape(x)[0]
        num_nodes = ops.shape(x)[1]
        C = self.k

        if mask is None:
            mask = ops.ones((batch_size, num_nodes, 1), dtype=x.dtype)
        else:
            mask = ops.cast(ops.reshape(mask, (batch_size, num_nodes, 1)), x.dtype)

        x = x * mask
        s = s * mask

        out = ops.selu(ops.matmul(ops.swapaxes(s, 1, 2), x))
        out_adj = ops.matmul(ops.matmul(ops.swapaxes(s, 1, 2), adj), s)

        # Spectral loss:
        degrees = ops.sum(adj, axis=-1, keepdims=True) * mask
        degrees_t = ops.swapaxes(degrees, 1, 2)

        m = ops.sum(degrees, axis=(1, 2)) / 2.0
        m_expand = ops.broadcast_to(ops.reshape(m, (-1, 1, 1)), (batch_size, C, C))

        ca = ops.matmul(ops.swapaxes(s, 1, 2), degrees)
        cb = ops.matmul(degrees_t, s)

        normalizer = ops.matmul(ca, cb) / 2.0 / m_expand
        decompose = out_adj - normalizer
        tr = ops.sum(ops.diagonal(decompose, axis1=1, axis2=2), axis=-1)
        spectral_loss = ops.mean(-tr / 2.0 / m)

        # Orthogonality regularization:
        ss = ops.matmul(ops.swapaxes(s, 1, 2), s)
        i_s = ops.eye(C, dtype=ss.dtype)
        norm_ss = ops.sqrt(ops.sum(ops.power(ss, 2), axis=(-1, -2), keepdims=True))
        norm_is = ops.sqrt(ops.cast(C, ss.dtype))
        diff = (ss / norm_ss) - (ops.expand_dims(i_s, axis=0) / norm_is)
        ortho_loss = ops.mean(ops.sqrt(ops.sum(ops.power(diff, 2), axis=(-1, -2))))

        # Cluster loss:
        cluster_size = ops.sum(s, axis=1)
        cluster_loss = ops.sqrt(ops.sum(ops.power(cluster_size, 2), axis=1))
        cluster_loss = ops.mean(cluster_loss / ops.sum(mask, axis=1) * norm_is - 1.0)

        # Fix and normalize coarsened adjacency matrix:
        eye_c = ops.expand_dims(ops.eye(C, dtype=out_adj.dtype), axis=0)
        out_adj = out_adj * (1.0 - eye_c)
        d = ops.sum(out_adj, axis=-1, keepdims=False)
        d = ops.expand_dims(ops.sqrt(d), axis=1) + 1e-15
        out_adj = (out_adj / d) / ops.swapaxes(d, 1, 2)

        return s, out, out_adj, spectral_loss, ortho_loss, cluster_loss

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.mlp.in_channels}, '
                f'num_clusters={self.mlp.out_channels})')

