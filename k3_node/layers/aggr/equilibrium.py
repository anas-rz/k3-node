from typing import List, Optional
from keras import layers, ops

from .base import Aggregation


class ResNetPotential(layers.Layer):
    def __init__(self, in_channels: int, out_channels: int, num_layers: List[int], **kwargs):
        super().__init__(**kwargs)
        sizes = [in_channels] + num_layers + [out_channels]
        self.layers_list = [
            layers.Dense(out_size, activation="tanh")
            for in_size, out_size in zip(sizes[:-2], sizes[1:-1])
        ]
        self.final_layer = layers.Dense(sizes[-1])
        self.res_trans = [
            layers.Dense(layer_size)
            for layer_size in num_layers + [out_channels]
        ]

    def reset_parameters(self):
        for l in self.layers_list:
            l.reset_parameters()
        self.final_layer.reset_parameters()
        for r in self.res_trans:
            r.reset_parameters()

    def call(self, inp):
        h = inp
        for layer, res in zip(self.layers_list + [self.final_layer], self.res_trans):
            h_next = layer(h)
            h = res(inp) + h_next
        return h


class EquilibriumAggregation(Aggregation):
    r"""The equilibrium aggregation layer from the `"Equilibrium Aggregation:
    Encoding Sets via Optimization" <https://arxiv.org/abs/2202.12795>`_ paper.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: List[int],
        grad_iter: int = 5,
        lamb: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.grad_iter = grad_iter
        self.initial_lamb = lamb

        self.potential = ResNetPotential(in_channels + out_channels, out_channels, num_layers)
        self.proj = layers.Dense(out_channels)

    def reset_parameters(self):
        self.potential.reset_parameters()
        self.proj.reset_parameters()

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        **kwargs,
    ):
        self.assert_index_present(index)
        index = ops.cast(index, dtype="int32")
        if dim_size is None:
            dim_size = ops.max(index) + 1

        # Initial mean aggregation as starting state
        x_mean = self.reduce(x, index, ptr, dim_size, dim, reduce="mean")
        y = self.proj(x_mean)

        # Unrolled iterative equilibrium updates
        for _ in range(self.grad_iter):
            y_expanded = ops.take(y, index, axis=0)
            inp = ops.concatenate([x, y_expanded], axis=-1)
            pot = self.potential(inp)
            pot_mean = self.reduce(pot, index, ptr, dim_size, dim, reduce="mean")
            y = y + 0.1 * pot_mean - self.initial_lamb * y

        return y

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

