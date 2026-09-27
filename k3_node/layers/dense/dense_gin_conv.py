from keras import initializers, layers, ops


class DenseGINConv(layers.Layer):
    r"""Applies the dense Graph Isomorphism Network (GIN) convolutional operator
    from the `"How Powerful are Graph Neural Networks?"
    <https://arxiv.org/abs/1810.00826>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathrm{MLP} \left( \left( \mathbf{A} + (1 + \epsilon)
        \cdot \mathbf{I} \right) \cdot \mathbf{X} \right)

    Args:
        nn (callable): A neural network layer (e.g. MLP or Dense) mapping feature
            representations to output representations.
        eps (float, optional): (Initial) :math:`\epsilon`-value. (default: :obj:`0.0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon` will
            be a trainable parameter. (default: :obj:`False`)

    Example:
        ```python
        import numpy as np
        import keras
        from k3_node.layers import DenseGINConv

        x = np.random.rand(2, 10, 8).astype("float32")  # batch of 2 graphs, 10 nodes, 8 features
        adj = (np.random.rand(2, 10, 10) > 0.7).astype("float32")  # dense adjacency matrices

        layer = DenseGINConv(keras.Sequential([keras.layers.Dense(16, activation="relu"), keras.layers.Dense(16)]))
        out = layer(x, adj)
        print(tuple(out.shape))  # (2, 10, 16)
        ```
    """
    def __init__(
        self,
        nn,
        eps: float = 0.0,
        train_eps: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        self.train_eps = train_eps

        self.eps = self.add_weight(
            shape=(1,),
            initializer=initializers.Constant(eps),
            trainable=train_eps,
            name="eps",
        )

    def reset_parameters(self):
        if hasattr(self.nn, 'reset_parameters'):
            self.nn.reset_parameters()
        self.eps.assign(ops.cast(ops.convert_to_tensor([self.initial_eps]), dtype=self.eps.dtype))

    def call(self, x, adj, mask=None, add_loop: bool = True):
        is_2d_x = (len(ops.shape(x)) == 2)
        if is_2d_x:
            x = ops.expand_dims(x, axis=0)
        if len(ops.shape(adj)) == 2:
            adj = ops.expand_dims(adj, axis=0)

        N = ops.shape(adj)[1]

        out = ops.matmul(adj, x)
        if add_loop:
            out = (1.0 + self.eps) * x + out

        out = self.nn(out)

        if mask is not None:
            out = out * ops.cast(ops.reshape(mask, (-1, N, 1)), x.dtype)

        if is_2d_x and ops.shape(out)[0] == 1:
            out = ops.squeeze(out, axis=0)

        return out

    def compute_output_shape(self, input_shape):
        if hasattr(self.nn, 'compute_output_shape'):
            return self.nn.compute_output_shape(input_shape)
        return input_shape

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

