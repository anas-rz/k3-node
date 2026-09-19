from typing import Optional, Union, Callable
import keras
from keras import ops, activations
from keras.layers import Layer

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.conv.gcn_conv import GCNConv


class AntiSymmetricConv(Layer):
    r"""The anti-symmetric graph convolutional operator from the
    `"Anti-Symmetric DGN: a Continuous approach to Deep Graph Neural Networks"
    <https://arxiv.org/abs/2202.13085>`_ paper.
    """
    def __init__(
        self,
        in_channels: int,
        phi: Optional[MessagePassing] = None,
        num_iters: int = 1,
        epsilon: float = 0.1,
        gamma: float = 0.1,
        act: Union[str, Callable, None] = "tanh",
        bias: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.num_iters = num_iters
        self.gamma = gamma
        self.epsilon = epsilon
        self.act = activations.get(act) if act is not None else None

        if phi is None:
            phi = GCNConv(in_channels, in_channels, bias=False)
        self.phi = phi

        self.W = self.add_weight(
            shape=(in_channels, in_channels),
            initializer="glorot_uniform",
            name="W",
        )

        if bias:
            self.bias = self.add_weight(
                shape=(in_channels,),
                initializer="zeros",
                name="bias",
            )
        else:
            self.bias = None

    def build(self, input_shape=None):
        if hasattr(self.phi, "build"):
            self.phi.build((None, self.in_channels))
        self.built = True

    def call(self, inputs, edge_index=None, **kwargs):
        if edge_index is None:
            if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                x, edge_index = inputs
            else:
                raise ValueError("Expected (x, edge_index) or x and edge_index")
        else:
            x = inputs

        eye = ops.eye(self.in_channels, dtype=self.W.dtype)
        antisymmetric_W = self.W - ops.transpose(self.W) - self.gamma * eye

        for _ in range(self.num_iters):
            h = self.phi(x, edge_index)
            h = ops.matmul(x, ops.transpose(antisymmetric_W)) + h

            if self.bias is not None:
                h = h + self.bias

            if self.act is not None:
                h = self.act(h)

            x = x + self.epsilon * h

        return x
