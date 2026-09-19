from typing import Optional

import keras
from keras import ops

from k3_node.layers.conv import SimpleConv
from k3_node.layers.norm import BatchNorm


class PMLP(keras.layers.Layer):
    r"""The P(ropagational)MLP model from the `"Graph Neural Networks are
    Inherently Good Generalizers: Insights by Bridging GNNs and MLPs"
    <https://arxiv.org/abs/2212.09034>`_ paper.

    :class:`PMLP` is identical to a standard MLP during training, but then
    adopts a GNN architecture during testing.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): The number of layers.
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        norm (bool, optional): If set to :obj:`False`, will not apply batch
            normalization. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the module will not
            learn additive biases. (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.,
        norm: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.use_norm = norm
        self.use_bias = bias
        # Instance-level training flag matching PyG's API: pmlp.training = False
        self.training = True

        # Build weight dimensions
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self._weight_shapes = [(dims[i], dims[i + 1]) for i in range(num_layers)]

        # We store weights as raw keras Variables so they are always available
        self._weights_list = []
        self._biases_list = []
        for i, (in_d, out_d) in enumerate(self._weight_shapes):
            w = self.add_weight(
                shape=(in_d, out_d),
                initializer=keras.initializers.GlorotUniform(),
                trainable=True,
                name=f"weight_{i}",
            )
            self._weights_list.append(w)
            if bias:
                b = self.add_weight(
                    shape=(out_d,),
                    initializer="zeros",
                    trainable=True,
                    name=f"bias_{i}",
                )
                self._biases_list.append(b)
            else:
                self._biases_list.append(None)

        self._norm_layer = None
        if norm:
            self._norm_layer = BatchNorm(
                hidden_channels,
                affine=False,
                track_running_stats=False,
            )

        self.conv = SimpleConv(aggr='mean', combine_root='self_loop')
        self._dropout = keras.layers.Dropout(dropout)

    def build(self, input_shape=None):
        self.built = True

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        for i, (in_d, out_d) in enumerate(self._weight_shapes):
            self._weights_list[i].assign(
                keras.initializers.GlorotUniform()(shape=(in_d, out_d))
            )
            if self.use_bias and self._biases_list[i] is not None:
                self._biases_list[i].assign(ops.zeros((out_d,)))

    def call(self, x, edge_index=None, training=None):
        """Forward pass.

        Args:
            x (Tensor): The node features of shape ``[N, in_channels]``.
            edge_index (Tensor, optional): The edge indices. Required during
                inference. (default: :obj:`None`)
            training (bool, optional): Override the instance-level
                ``self.training`` flag. (default: :obj:`None`)
        """
        # Respect both call-time kwarg and instance-level flag (PyG compat)
        is_training = training if training is not None else self.training

        if not is_training and edge_index is None:
            raise ValueError(
                f"'edge_index' needs to be present during inference "
                f"in '{self.__class__.__name__}'"
            )

        for i in range(self.num_layers):
            # Apply weight multiplication (like x @ W^T where W is [in, out])
            x = x @ self._weights_list[i]

            if not is_training:
                x = self.conv(x, edge_index)

            if self.use_bias and self._biases_list[i] is not None:
                x = x + self._biases_list[i]

            if i != self.num_layers - 1:
                if self._norm_layer is not None:
                    x = self._norm_layer(x, training=is_training)
                x = ops.relu(x)
                x = self._dropout(x, training=is_training)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, num_layers={self.num_layers})')
