from functools import partial
from typing import Callable, Optional, Union, Tuple
from collections import defaultdict
import numpy as np
import sympy as sym
import keras
from keras import ops

from k3_node.layers.aggr import SumAggregation
from k3_node.layers.pool import radius_graph, global_add_pool
from k3_node.models.dimenet_utils import bessel_basis, real_sph_harm


def triplets(
    edge_index,
    num_nodes: Optional[int] = None,
) -> Tuple[any, any, any, any, any, any, any]:
    r"""Extracts triplet indices (k -> j -> i) from edge_index."""
    edge_index_np = ops.convert_to_numpy(edge_index)
    row, col = edge_index_np[0], edge_index_np[1]
    E = len(row)

    edges_by_target = defaultdict(list)
    for e_kj, target in enumerate(col):
        edges_by_target[int(target)].append(e_kj)

    idx_i_list = []
    idx_j_list = []
    idx_k_list = []
    idx_kj_list = []
    idx_ji_list = []

    for e_ji in range(E):
        j = int(row[e_ji])
        i = int(col[e_ji])
        for e_kj in edges_by_target.get(j, []):
            k = int(row[e_kj])
            if k != i:
                idx_i_list.append(i)
                idx_j_list.append(j)
                idx_k_list.append(k)
                idx_kj_list.append(e_kj)
                idx_ji_list.append(e_ji)

    device_dtype = "int32"
    return (
        ops.cast(ops.convert_to_tensor(col), device_dtype),
        ops.cast(ops.convert_to_tensor(row), device_dtype),
        ops.cast(ops.convert_to_tensor(np.array(idx_i_list, dtype=np.int32)), device_dtype),
        ops.cast(ops.convert_to_tensor(np.array(idx_j_list, dtype=np.int32)), device_dtype),
        ops.cast(ops.convert_to_tensor(np.array(idx_k_list, dtype=np.int32)), device_dtype),
        ops.cast(ops.convert_to_tensor(np.array(idx_kj_list, dtype=np.int32)), device_dtype),
        ops.cast(ops.convert_to_tensor(np.array(idx_ji_list, dtype=np.int32)), device_dtype),
    )


def cross_product(a, b):
    r"""Cross product of 3D vectors along the last axis."""
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    return ops.stack([
        a1 * b2 - a2 * b1,
        a2 * b0 - a0 * b2,
        a0 * b1 - a1 * b0,
    ], axis=-1)


class Envelope(keras.layers.Layer):
    def __init__(self, exponent: int, **kwargs):
        super().__init__(**kwargs)
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def call(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = ops.power(x, p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        env = (1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2)
        mask = ops.cast(x < 1.0, x.dtype)
        return env * mask


class BesselBasisLayer(keras.layers.Layer):
    """Layer ``BesselBasisLayer``.

    Example:
        ```python
        import numpy as np
        from k3_node.models import BesselBasisLayer

        dist = np.array([0.9, 1.5, 3.2], dtype="float32")  # interatomic distances
        rbf = BesselBasisLayer(num_radial=6, cutoff=5.0)(dist)  # radial basis expansion
        print(tuple(rbf.shape))  # (3, 6)
        ```
    """
    def __init__(self, num_radial: int, cutoff: float = 5.0, envelope_exponent: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        freq = np.arange(1, num_radial + 1, dtype=np.float32) * np.pi
        self.freq = self.add_weight(
            name="freq",
            shape=(num_radial,),
            initializer=keras.initializers.Constant(freq),
            trainable=True,
            dtype="float32",
        )

    def call(self, dist):
        dist = ops.expand_dims(dist, -1) / self.cutoff
        return self.envelope(dist) * ops.sin(self.freq * dist)


class SphericalBasisLayer(keras.layers.Layer):
    """Layer ``SphericalBasisLayer``.

    Example:
        ```python
        import numpy as np
        from k3_node.models import SphericalBasisLayer

        dist = np.array([0.9, 1.5, 3.2], dtype="float32")  # edge lengths
        angle = np.array([1.2, 2.0], dtype="float32")  # angles of two edge triplets
        idx_kj = np.array([0, 2])  # edge (k -> j) of each triplet
        sbf = SphericalBasisLayer(num_spherical=3, num_radial=6, cutoff=5.0)(dist, angle, idx_kj)
        print(tuple(sbf.shape))  # (2, 18): num_spherical * num_radial features per triplet
        ```
    """
    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        envelope_exponent: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols("x theta")
        modules = {"sin": ops.sin, "cos": ops.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = float(sym.lambdify([theta], sph_harm_forms[i][0], modules)(0))
                self.sph_funcs.append(partial(self._sph_to_tensor, sph1))
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    @staticmethod
    def _sph_to_tensor(sph, x):
        return ops.zeros_like(x) + sph

    def call(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf_list = [f(dist) for f in self.bessel_funcs]
        rbf = ops.stack(rbf_list, axis=1)
        rbf = ops.expand_dims(self.envelope(dist), -1) * rbf

        cbf_list = [f(angle) for f in self.sph_funcs]
        cbf = ops.stack(cbf_list, axis=1)

        n, k = self.num_spherical, self.num_radial
        rbf_kj = ops.take(rbf, idx_kj, axis=0)
        rbf_kj = ops.reshape(rbf_kj, (-1, n, k))
        cbf_mat = ops.reshape(cbf, (-1, n, 1))
        out = ops.reshape(rbf_kj * cbf_mat, (-1, n * k))
        return out


class EmbeddingBlock(keras.layers.Layer):
    def __init__(self, num_radial: int, hidden_channels: int, act: Callable, **kwargs):
        super().__init__(**kwargs)
        self.act = act
        self.emb = keras.layers.Embedding(95, hidden_channels)
        self.lin_rbf = keras.layers.Dense(hidden_channels)
        self.lin = keras.layers.Dense(hidden_channels)

    def call(self, x, rbf, i, j):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        x_i = ops.take(x, i, axis=0)
        x_j = ops.take(x, j, axis=0)
        concat = ops.concatenate([x_i, x_j, rbf], axis=-1)
        return self.act(self.lin(concat))


class ResidualLayer(keras.layers.Layer):
    def __init__(self, hidden_channels: int, act: Callable, **kwargs):
        super().__init__(**kwargs)
        self.act = act
        self.lin1 = keras.layers.Dense(hidden_channels)
        self.lin2 = keras.layers.Dense(hidden_channels)

    def build(self, input_shape=None):
        self.built = True

    def call(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class InteractionBlock(keras.layers.Layer):
    def __init__(
        self,
        hidden_channels: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.act = act
        self.hidden_channels = hidden_channels
        self.num_bilinear = num_bilinear

        self.lin_rbf = keras.layers.Dense(hidden_channels, use_bias=False)
        self.lin_sbf = keras.layers.Dense(num_bilinear, use_bias=False)

        self.lin_kj = keras.layers.Dense(hidden_channels)
        self.lin_ji = keras.layers.Dense(hidden_channels)

        self.W = self.add_weight(
            name="W",
            shape=(hidden_channels, num_bilinear, hidden_channels),
            initializer=keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2.0 / hidden_channels)),
            trainable=True,
            dtype="float32",
        )

        self.layers_before_skip = [
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ]
        self.lin = keras.layers.Dense(hidden_channels)
        self.layers_after_skip = [
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ]
        self.sum_aggr = SumAggregation()

    def build(self, input_shape=None):
        self.built = True

    def call(self, x, rbf, sbf, idx_kj, idx_ji):
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf

        # einsum("wj,wl,ijl->wi", sbf, x_kj[idx_kj], W), one bilinear channel j at a time (no
        # [triplets, num_bilinear, hidden] intermediate). Each edge is multiplied with W_j once and
        # then gathered to its triplets: the same values, with far fewer multiplications, since a
        # molecule has many more triplets than edges.
        x_kj = sum(sbf[:, j:j + 1] * ops.take(ops.matmul(x_kj, ops.transpose(self.W[:, j, :])), idx_kj, axis=0)
                   for j in range(self.num_bilinear))
        x_kj = self.sum_aggr(x_kj, index=idx_ji, dim_size=ops.shape(x)[0])

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class InteractionPPBlock(keras.layers.Layer):
    def __init__(
        self,
        hidden_channels: int,
        int_emb_size: int,
        basis_emb_size: int,
        num_spherical: int,
        num_radial: int,
        num_before_skip: int,
        num_after_skip: int,
        act: Callable,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.act = act

        self.lin_rbf1 = keras.layers.Dense(basis_emb_size, use_bias=False)
        self.lin_rbf2 = keras.layers.Dense(hidden_channels, use_bias=False)

        self.lin_sbf1 = keras.layers.Dense(basis_emb_size, use_bias=False)
        self.lin_sbf2 = keras.layers.Dense(int_emb_size, use_bias=False)

        self.lin_kj = keras.layers.Dense(hidden_channels)
        self.lin_ji = keras.layers.Dense(hidden_channels)

        self.lin_down = keras.layers.Dense(int_emb_size, use_bias=False)
        self.lin_up = keras.layers.Dense(hidden_channels, use_bias=False)

        self.layers_before_skip = [
            ResidualLayer(hidden_channels, act) for _ in range(num_before_skip)
        ]
        self.lin = keras.layers.Dense(hidden_channels)
        self.layers_after_skip = [
            ResidualLayer(hidden_channels, act) for _ in range(num_after_skip)
        ]
        self.sum_aggr = SumAggregation()

    def build(self, input_shape=None):
        self.built = True

    def call(self, x, rbf, sbf, idx_kj, idx_ji):
        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))

        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = ops.take(x_kj, idx_kj, axis=0) * sbf

        x_kj = self.sum_aggr(x_kj, index=idx_ji, dim_size=ops.shape(x)[0])
        x_kj = self.act(self.lin_up(x_kj))

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class OutputBlock(keras.layers.Layer):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = "zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.act = act
        self.lin_rbf = keras.layers.Dense(hidden_channels, use_bias=False)
        self.lins = [
            keras.layers.Dense(hidden_channels) for _ in range(num_layers)
        ]
        self.lin = keras.layers.Dense(
            out_channels,
            use_bias=False,
            kernel_initializer=output_initializer,
        )
        self.sum_aggr = SumAggregation()

    def build(self, input_shape=None):
        self.built = True

    def call(self, x, rbf, i, num_nodes: Optional[int] = None):
        x = self.lin_rbf(rbf) * x
        x = self.sum_aggr(x, index=i, dim_size=num_nodes)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class OutputPPBlock(keras.layers.Layer):
    def __init__(
        self,
        num_radial: int,
        hidden_channels: int,
        out_emb_channels: int,
        out_channels: int,
        num_layers: int,
        act: Callable,
        output_initializer: str = "zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.act = act
        self.lin_rbf = keras.layers.Dense(hidden_channels, use_bias=False)
        self.lin_up = keras.layers.Dense(out_emb_channels, use_bias=False)
        self.lins = [
            keras.layers.Dense(out_emb_channels) for _ in range(num_layers)
        ]
        self.lin = keras.layers.Dense(
            out_channels,
            use_bias=False,
            kernel_initializer=output_initializer,
        )
        self.sum_aggr = SumAggregation()

    def build(self, input_shape=None):
        self.built = True

    def call(self, x, rbf, i, num_nodes: Optional[int] = None):
        x = self.lin_rbf(rbf) * x
        x = self.sum_aggr(x, index=i, dim_size=num_nodes)
        x = self.lin_up(x)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)


class DimeNet(keras.layers.Layer):
    r"""The directional message passing neural network (DimeNet) from the
    `"Directional Message Passing for Molecular Graphs"
    <https://arxiv.org/abs/2003.03123>`_ paper.

    Example:
        ```python
        import numpy as np
        from k3_node.models import DimeNet

        z = np.array([6, 8, 1, 1, 1])  # atomic numbers of a small molecule
        pos = np.random.rand(5, 3).astype("float32") * 2.0  # 3D coordinates (Angstrom)
        batch = np.array([0, 0, 0, 1, 1])  # two molecules: atoms 0-2 and atoms 3-4

        model = DimeNet(hidden_channels=16, out_channels=1, num_blocks=2, num_bilinear=8,
                        num_spherical=3, num_radial=6, cutoff=5.0)
        energy = model(z, pos, batch=batch)  # one prediction per molecule
        print(tuple(energy.shape))  # (2, 1)
        ```
    """
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = "swish",
        output_initializer: str = "zeros",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")

        self.act = keras.activations.get(act) if isinstance(act, str) else act
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, self.act)

        self.output_blocks = [
            OutputBlock(
                num_radial,
                hidden_channels,
                out_channels,
                num_output_layers,
                self.act,
                output_initializer,
            )
            for _ in range(num_blocks + 1)
        ]

        self.interaction_blocks = [
            InteractionBlock(
                hidden_channels,
                num_bilinear,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                self.act,
            )
            for _ in range(num_blocks)
        ]

    def call(self, z, pos, batch=None, batch_size=None):
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=self.max_num_neighbors,
        )

        num_nodes = ops.shape(z)[0]
        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(edge_index, num_nodes=num_nodes)

        pos_i = ops.take(pos, i, axis=0)
        pos_j = ops.take(pos, j, axis=0)
        dist = ops.sqrt(ops.sum(ops.power(pos_i - pos_j, 2), axis=-1))

        if isinstance(self, DimeNetPlusPlus):
            pos_k = ops.take(pos, idx_k, axis=0)
            pos_j_triplet = ops.take(pos, idx_j, axis=0)
            pos_i_triplet = ops.take(pos, idx_i, axis=0)
            pos_jk = pos_j_triplet - pos_k
            pos_ij = pos_i_triplet - pos_j_triplet
            a = ops.sum(pos_ij * pos_jk, axis=-1)
            b = ops.sqrt(ops.sum(ops.power(cross_product(pos_ij, pos_jk), 2), axis=-1))
        else:
            pos_k = ops.take(pos, idx_k, axis=0)
            pos_j_triplet = ops.take(pos, idx_j, axis=0)
            pos_i_triplet = ops.take(pos, idx_i, axis=0)
            pos_ji = pos_j_triplet - pos_i_triplet
            pos_ki = pos_k - pos_i_triplet
            a = ops.sum(pos_ji * pos_ki, axis=-1)
            b = ops.sqrt(ops.sum(ops.power(cross_product(pos_ji, pos_ki), 2), axis=-1))

        angle = ops.arctan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        z = ops.cast(z, "int32")
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=num_nodes)

        for interaction_block, output_block in zip(self.interaction_blocks, self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P = P + output_block(x, rbf, i, num_nodes=num_nodes)

        if batch is None:
            return ops.sum(P, axis=0)
        else:
            batch = ops.cast(batch, "int32")
            return global_add_pool(P, batch, size=batch_size)


class DimeNetPlusPlus(DimeNet):
    r"""The DimeNet++ from the `"Fast and Uncertainty-Aware
    Directional Message Passing for Non-Equilibrium Molecules"
    <https://arxiv.org/abs/2011.14115>`_ paper.

    Example:
        ```python
        import numpy as np
        from k3_node.models import DimeNetPlusPlus

        z = np.array([6, 8, 1, 1, 1])  # atomic numbers of a small molecule
        pos = np.random.rand(5, 3).astype("float32") * 2.0  # 3D coordinates (Angstrom)
        batch = np.array([0, 0, 0, 1, 1])  # two molecules: atoms 0-2 and atoms 3-4

        model = DimeNetPlusPlus(hidden_channels=16, out_channels=1, num_blocks=2, int_emb_size=8,
                                basis_emb_size=8, out_emb_channels=16, num_spherical=3, num_radial=6, cutoff=5.0)
        energy = model(z, pos, batch=batch)
        print(tuple(energy.shape))  # (2, 1)
        ```
    """
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        envelope_exponent: int = 5,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_output_layers: int = 3,
        act: Union[str, Callable] = "swish",
        output_initializer: str = "zeros",
        **kwargs,
    ):
        super(DimeNet, self).__init__(**kwargs)
        if num_spherical < 2:
            raise ValueError("'num_spherical' should be greater than 1")

        self.act = keras.activations.get(act) if isinstance(act, str) else act
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff, envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, self.act)

        self.output_blocks = [
            OutputPPBlock(
                num_radial,
                hidden_channels,
                out_emb_channels,
                out_channels,
                num_output_layers,
                self.act,
                output_initializer,
            )
            for _ in range(num_blocks + 1)
        ]

        self.interaction_blocks = [
            InteractionPPBlock(
                hidden_channels,
                int_emb_size,
                basis_emb_size,
                num_spherical,
                num_radial,
                num_before_skip,
                num_after_skip,
                self.act,
            )
            for _ in range(num_blocks)
        ]



QM9_TARGETS = {0: 'mu', 1: 'alpha', 2: 'homo', 3: 'lumo', 5: 'r2', 6: 'zpve', 7: 'U0', 8: 'U', 9: 'H',
               10: 'G', 11: 'Cv'}


def _load_qm9_pretrained(cls, root, dataset, target, url, folder, config):
    """Builds ``cls(**config)`` and copies the official TensorFlow DimeNet(++) checkpoint for
    ``target`` into it; returns the model and PyG's train/val/test split of ``dataset``."""
    import os
    import os.path as osp

    import tensorflow as tf

    from k3_node.data.download import download_url

    assert 0 <= target <= 11 and target != 4, "no pre-trained model for this target"
    path = osp.join(osp.expanduser(root), folder, QM9_TARGETS[target])
    os.makedirs(path, exist_ok=True)
    if not osp.exists(osp.join(path, 'checkpoint')):
        for name in ['checkpoint', 'ckpt.data-00000-of-00002', 'ckpt.data-00001-of-00002', 'ckpt.index']:
            download_url(f'{url}/{QM9_TARGETS[target]}/{name}', path)
    reader = tf.train.load_checkpoint(osp.join(path, 'ckpt'))

    model = cls(**config)
    model(np.array([6, 1, 1, 1, 1]), np.array([[0, 0, 0], [0.6, 0.6, 0.6], [-0.6, -0.6, 0.6], [-0.6, 0.6, -0.6],
                                              [0.6, -0.6, -0.6]], dtype="float32"))  # create the weights

    def copy_(variable, name):  # Keras kernels already use TensorFlow's (in, out) layout
        variable.assign(reader.get_tensor(f'{name}/.ATTRIBUTES/VARIABLE_VALUE'))

    def copy_dense(layer, name, bias=True):
        copy_(layer.kernel, f'{name}/kernel')
        if bias:
            copy_(layer.bias, f'{name}/bias')

    def copy_residuals(layers, name):
        for j, layer in enumerate(layers):
            copy_dense(layer.lin1, f'{name}/{j}/dense_1')
            copy_dense(layer.lin2, f'{name}/{j}/dense_2')

    plus = cls is DimeNetPlusPlus
    copy_(model.rbf.freq, 'rbf_layer/frequencies')
    copy_(model.emb.emb.embeddings, 'emb_block/embeddings')
    copy_dense(model.emb.lin_rbf, 'emb_block/dense_rbf')
    copy_dense(model.emb.lin, 'emb_block/dense')
    for i, block in enumerate(model.output_blocks):
        copy_dense(block.lin_rbf, f'output_blocks/{i}/dense_rbf', bias=False)
        if plus:
            copy_dense(block.lin_up, f'output_blocks/{i}/up_projection', bias=False)
        for j, lin in enumerate(block.lins):
            copy_dense(lin, f'output_blocks/{i}/dense_layers/{j}')
        copy_dense(block.lin, f'output_blocks/{i}/dense_final', bias=False)
    for i, block in enumerate(model.interaction_blocks):
        name = f'int_blocks/{i}'
        if plus:
            for layer in ['rbf1', 'rbf2', 'sbf1', 'sbf2']:
                copy_dense(getattr(block, f'lin_{layer}'), f'{name}/dense_{layer}', bias=False)
            copy_dense(block.lin_down, f'{name}/down_projection', bias=False)
            copy_dense(block.lin_up, f'{name}/up_projection', bias=False)
        else:
            copy_dense(block.lin_rbf, f'{name}/dense_rbf', bias=False)
            copy_dense(block.lin_sbf, f'{name}/dense_sbf', bias=False)
            copy_(block.W, f'{name}/bilinear')
        copy_dense(block.lin_kj, f'{name}/dense_kj')
        copy_dense(block.lin_ji, f'{name}/dense_ji')
        copy_residuals(block.layers_before_skip, f'{name}/layers_before_skip')
        copy_dense(block.lin, f'{name}/final_before_skip')
        copy_residuals(block.layers_after_skip, f'{name}/layers_after_skip')

    # The split of the official DimeNet implementation
    perm = np.random.RandomState(seed=42).permutation(np.arange(130831))
    return model, (dataset[perm[:110000]], dataset[perm[110000:120000]], dataset[perm[120000:]])


def _dimenet_from_qm9_pretrained(cls, root: str, dataset, target: int):
    r"""Returns a :class:`DimeNet` pre-trained on QM9 target ``target`` (the official weights of
    the DimeNet authors, read with TensorFlow), and the train/validation/test split it used.
    ``dataset`` is a :class:`~k3_node.datasets.QM9` whose targets were reordered as in PyG's
    ``qm9_pretrained_dimenet`` example."""
    return _load_qm9_pretrained(
        cls, root, dataset, target, 'https://github.com/klicperajo/dimenet/raw/master/pretrained/dimenet',
        'pretrained_dimenet', dict(hidden_channels=128, out_channels=1, num_blocks=6, num_bilinear=8,
                                   num_spherical=7, num_radial=6, cutoff=5.0, envelope_exponent=5,
                                   num_before_skip=1, num_after_skip=2, num_output_layers=3))


def _dimenet_pp_from_qm9_pretrained(cls, root: str, dataset, target: int):
    r"""Returns a :class:`DimeNetPlusPlus` pre-trained on QM9 target ``target`` and its split (see
    :meth:`DimeNet.from_qm9_pretrained`)."""
    return _load_qm9_pretrained(
        cls, root, dataset, target, 'https://raw.githubusercontent.com/gasteigerjo/dimenet/master/pretrained/dimenet_pp',
        'pretrained_dimenet_pp', dict(hidden_channels=128, out_channels=1, num_blocks=4, int_emb_size=64,
                                      basis_emb_size=8, out_emb_channels=256, num_spherical=7, num_radial=6,
                                      cutoff=5.0, max_num_neighbors=32, envelope_exponent=5, num_before_skip=1,
                                      num_after_skip=2, num_output_layers=3))


DimeNet.from_qm9_pretrained = classmethod(_dimenet_from_qm9_pretrained)
DimeNetPlusPlus.from_qm9_pretrained = classmethod(_dimenet_pp_from_qm9_pretrained)
