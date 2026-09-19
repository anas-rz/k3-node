import numpy as np
import keras
from keras import ops
from typing import Optional, Callable

from k3_node.layers.conv.message_passing import MessagePassing
from k3_node.layers.pool import radius_graph, global_add_pool, global_mean_pool


DEFAULT_ATOMIC_MASSES = [
    0.0, 1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998,
    20.180, 22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.95,
    39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845, 58.933,
    58.693, 63.546, 65.38, 69.723, 72.630, 74.922, 78.971, 79.904, 83.798,
    85.468, 87.62, 88.906, 91.224, 92.906, 95.95, 98.0, 101.07, 102.91,
    106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.60, 126.90, 131.29,
    132.91, 137.33, 138.91, 140.12, 140.91, 144.24, 145.0, 150.36, 151.96,
    157.25, 158.93, 162.50, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49,
    180.95, 183.84, 186.21, 190.23, 192.22, 195.08, 196.97, 200.59, 204.38,
    207.2, 208.98, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.04,
    231.04, 238.03, 237.0, 244.0, 243.0, 247.0, 247.0, 251.0, 252.0,
]


class ShiftedSoftplus(keras.layers.Layer):
    r"""Shifted softplus activation function: :math:`\ln(1 + e^x) - \ln(2)`."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shift = float(np.log(2.0))

    def call(self, x):
        return ops.softplus(x) - self.shift


class GaussianSmearing(keras.layers.Layer):
    r"""Smears interatomic distances using Gaussian basis functions."""
    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_gaussians: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.start = start
        self.stop = stop
        self.num_gaussians = num_gaussians

        offset = np.linspace(start, stop, num_gaussians, dtype=np.float32)
        diff = float(offset[1] - offset[0])
        self.coeff = -0.5 / (diff ** 2)
        self.offset = self.add_weight(
            name="offset",
            shape=(num_gaussians,),
            initializer=keras.initializers.Constant(offset),
            trainable=False,
            dtype="float32",
        )

    def call(self, dist):
        dist = ops.expand_dims(dist, -1) - ops.expand_dims(self.offset, 0)
        return ops.exp(self.coeff * ops.power(dist, 2))


class RadiusInteractionGraph(keras.layers.Layer):
    r"""Creates edges based on atom positions :obj:`pos` to all points within
    the cutoff distance.
    """
    def __init__(self, cutoff: float = 10.0, max_num_neighbors: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors

    def call(self, pos, batch=None):
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )
        row = edge_index[0]
        col = edge_index[1]
        pos_row = ops.take(pos, row, axis=0)
        pos_col = ops.take(pos, col, axis=0)
        edge_weight = ops.sqrt(ops.sum(ops.power(pos_row - pos_col, 2), axis=-1))
        return edge_index, edge_weight


class CFConv(MessagePassing):
    r"""Continuous-filter convolution layer."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_filters: int,
        nn: keras.layers.Layer,
        cutoff: float,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.nn = nn
        self.cutoff = cutoff
        self.lin1 = keras.layers.Dense(num_filters, use_bias=False)
        self.lin2 = keras.layers.Dense(out_channels, use_bias=True)

    def build(self, input_shape=None):
        self.lin1.build((None, self.in_channels))
        self.lin2.build((None, self.num_filters))
        super().build(input_shape)

    def call(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (ops.cos(edge_weight * np.pi / self.cutoff) + 1.0)
        W = self.nn(edge_attr) * ops.expand_dims(C, -1)
        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class InteractionBlock(keras.layers.Layer):
    r"""Interaction block used in SchNet."""
    def __init__(
        self,
        hidden_channels: int,
        num_gaussians: int,
        num_filters: int,
        cutoff: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.num_gaussians = num_gaussians
        self.num_filters = num_filters
        self.cutoff = cutoff

        self.mlp = keras.Sequential([
            keras.layers.Dense(num_filters),
            ShiftedSoftplus(),
            keras.layers.Dense(num_filters),
        ])
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters, self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = keras.layers.Dense(hidden_channels)

    def call(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class SchNet(keras.layers.Layer):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper.

    Args:
        hidden_channels (int, optional): Hidden embedding size. (default: 128)
        num_filters (int, optional): The number of filters to use. (default: 128)
        num_interactions (int, optional): The number of interaction blocks. (default: 6)
        num_gaussians (int, optional): The number of gaussians. (default: 50)
        cutoff (float, optional): Cutoff distance. (default: 10.0)
        interaction_graph (callable, optional): Interaction graph builder. (default: None)
        max_num_neighbors (int, optional): Maximum neighbors per atom. (default: 32)
        readout (str, optional): Readout pooling (add, sum, mean). (default: "add")
        dipole (bool, optional): Predict dipole moment magnitude. (default: False)
        mean (float, optional): Mean of target property. (default: None)
        std (float, optional): Standard deviation of target property. (default: None)
        atomref (tensor, optional): Reference atomic values. (default: None)
    """
    def __init__(
        self,
        hidden_channels: int = 128,
        num_filters: int = 128,
        num_interactions: int = 6,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        interaction_graph: Optional[Callable] = None,
        max_num_neighbors: int = 32,
        readout: str = "add",
        dipole: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        atomref: Optional[any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.readout = readout
        self.dipole = dipole
        self.mean = mean
        self.std = std
        self.scale = None

        try:
            import ase
            masses = np.array(ase.data.atomic_masses, dtype=np.float32)
        except ImportError:
            masses = np.array(DEFAULT_ATOMIC_MASSES, dtype=np.float32)

        self.atomic_mass = self.add_weight(
            name="atomic_mass",
            shape=(len(masses),),
            initializer=keras.initializers.Constant(masses),
            trainable=False,
            dtype="float32",
        )

        self.embedding = keras.layers.Embedding(100, hidden_channels)

        if interaction_graph is not None:
            self.interaction_graph = interaction_graph
        else:
            self.interaction_graph = RadiusInteractionGraph(cutoff, max_num_neighbors)

        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = [
            InteractionBlock(hidden_channels, num_gaussians, num_filters, cutoff)
            for _ in range(num_interactions)
        ]

        self.lin1 = keras.layers.Dense(hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = keras.layers.Dense(1)

        self.has_atomref = atomref is not None
        if atomref is not None:
            self.atomref = keras.layers.Embedding(
                100,
                1,
                embeddings_initializer=keras.initializers.Constant(atomref),
            )
        else:
            self.atomref = None

    def call(self, z, pos, batch=None):
        if batch is None:
            batch = ops.zeros(ops.shape(z), dtype="int32")
        else:
            batch = ops.cast(batch, "int32")

        z = ops.cast(z, "int32")
        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            mass = ops.take(self.atomic_mass, z, axis=0)
            mass = ops.expand_dims(mass, -1)
            M = global_add_pool(mass, batch)
            c = global_add_pool(mass * pos, batch) / (M + 1e-8)
            c_per_atom = ops.take(c, batch, axis=0)
            h = h * (pos - c_per_atom)

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        if self.dipole or self.readout in ["add", "sum"]:
            out = global_add_pool(h, batch)
        else:
            out = global_mean_pool(h, batch)

        if self.dipole:
            out = ops.sqrt(ops.sum(ops.power(out, 2), axis=-1, keepdims=True))

        if self.scale is not None:
            out = self.scale * out

        return out
