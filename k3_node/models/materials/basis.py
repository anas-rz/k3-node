"""Radial, angular, and geometric basis functions for materials models."""

from __future__ import annotations

import math
from typing import Optional, Union, Tuple, Literal
import numpy as np
import keras
from keras import layers, ops


def polynomial_cutoff(r, cutoff: float, exponent: int = 3):
    """Envelope polynomial function that ensures a smooth cutoff."""
    coef1 = -(exponent + 1) * (exponent + 2) / 2
    coef2 = exponent * (exponent + 2)
    coef3 = -exponent * (exponent + 1) / 2
    ratio = r / cutoff
    poly = 1.0 + coef1 * (ratio ** exponent) + coef2 * (ratio ** (exponent + 1)) + coef3 * (ratio ** (exponent + 2))
    return ops.where(r <= cutoff, poly, 0.0)


def cosine_cutoff(r, cutoff: float):
    """Cosine cutoff function."""
    return ops.where(r <= cutoff, 0.5 * (ops.cos(math.pi * r / cutoff) + 1.0), 0.0)


def compute_pair_vector_and_distance(pos, edge_index, pbc_offshift=None):
    """Compute pair displacement vectors and pairwise Euclidean distances.
    
    Args:
        pos: Atomic positions of shape [num_nodes, 3].
        edge_index: COO edge indices of shape [2, num_edges].
        pbc_offshift: Periodic boundary condition offset displacement of shape [num_edges, 3] or None.
        
    Returns:
        pair_vectors: Displacement vectors of shape [num_edges, 3].
        bond_dists: Euclidean distances of shape [num_edges].
    """
    src = ops.cast(edge_index[0], "int32")
    dst = ops.cast(edge_index[1], "int32")
    num_nodes = ops.shape(pos)[0]
    src = ops.clip(src, 0, ops.maximum(num_nodes - 1, 0))
    dst = ops.clip(dst, 0, ops.maximum(num_nodes - 1, 0))
    pos_src = ops.take(pos, src, axis=0)
    pos_dst = ops.take(pos, dst, axis=0)
    vec = pos_dst - pos_src
    if pbc_offshift is not None:
        vec = vec + pbc_offshift
    dist = ops.sqrt(ops.maximum(ops.sum(vec ** 2, axis=-1), 1e-12))
    return vec, dist


def compute_theta(pos, edge_index, line_edge_index, pbc_offshift=None):
    """Compute bond angles for triplets in directed line graph."""
    vec, _ = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
    src_bond = ops.cast(line_edge_index[0], "int32")
    dst_bond = ops.cast(line_edge_index[1], "int32")
    num_edges = ops.shape(vec)[0]
    src_bond = ops.clip(src_bond, 0, ops.maximum(num_edges - 1, 0))
    dst_bond = ops.clip(dst_bond, 0, ops.maximum(num_edges - 1, 0))
    vec1 = ops.take(vec, src_bond, axis=0)
    vec2 = ops.take(vec, dst_bond, axis=0)
    norm1 = ops.sqrt(ops.maximum(ops.sum(vec1 ** 2, axis=-1, keepdims=True), 1e-12))
    norm2 = ops.sqrt(ops.maximum(ops.sum(vec2 ** 2, axis=-1, keepdims=True), 1e-12))
    cos_theta = ops.sum(vec1 * vec2, axis=-1) / ops.squeeze(norm1 * norm2, axis=-1)
    cos_theta = ops.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    return ops.arccos(cos_theta)


def compute_theta_and_phi(pos, edge_index, line_edge_index, pbc_offshift=None):
    """Compute theta (azimuthal) and phi (polar) angles for triplets."""
    vec, _ = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
    src_bond = ops.cast(line_edge_index[0], "int32")
    dst_bond = ops.cast(line_edge_index[1], "int32")
    num_edges = ops.shape(vec)[0]
    src_bond = ops.clip(src_bond, 0, ops.maximum(num_edges - 1, 0))
    dst_bond = ops.clip(dst_bond, 0, ops.maximum(num_edges - 1, 0))
    vec1 = ops.take(vec, src_bond, axis=0)
    vec2 = ops.take(vec, dst_bond, axis=0)
    norm1 = ops.sqrt(ops.maximum(ops.sum(vec1 ** 2, axis=-1, keepdims=True), 1e-12))
    norm2 = ops.sqrt(ops.maximum(ops.sum(vec2 ** 2, axis=-1, keepdims=True), 1e-12))
    cos_theta = ops.sum(vec1 * vec2, axis=-1) / ops.squeeze(norm1 * norm2, axis=-1)
    cos_theta = ops.clip(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = ops.arccos(cos_theta)
    # cross product of bond vectors for phi
    cross = ops.cross(vec1, vec2)
    phi = ops.arctan2(cross[..., 1], cross[..., 0])
    return theta, phi


class GaussianExpansion(layers.Layer):
    """Gaussian Radial Basis Function expansion."""

    def __init__(
        self,
        initial: float = 0.0,
        final: float = 5.0,
        num_centers: int = 20,
        width: Optional[float] = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initial = initial
        self.final = final
        self.num_centers = num_centers
        self.user_width = width

    def build(self, input_shape=None):
        centers_np = np.linspace(self.initial, self.final, self.num_centers, dtype=np.float32)
        if self.user_width is None:
            width_val = float(1.0 / np.diff(centers_np).mean())
        else:
            width_val = float(self.user_width)

        self.centers = self.add_weight(
            name="centers",
            shape=(self.num_centers,),
            initializer=keras.initializers.Constant(centers_np),
            trainable=False,
            dtype="float32",
        )
        self.width = self.add_weight(
            name="width",
            shape=(),
            initializer=keras.initializers.Constant(width_val),
            trainable=False,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, bond_dists):
        # [num_edges, 1] - [1, num_centers]
        diff = ops.expand_dims(bond_dists, axis=-1) - ops.expand_dims(self.centers, axis=0)
        return ops.exp(-self.width * (diff ** 2))


class RadialBesselFunction(layers.Layer):
    """Zeroth-order spherical Bessel function radial basis with optional learnable roots."""

    def __init__(
        self,
        max_n: int = 3,
        cutoff: float = 5.0,
        learnable: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_n = max_n
        self.cutoff = float(cutoff)
        self.learnable = learnable
        self.inv_cutoff = 1.0 / self.cutoff
        self.norm_const = float(math.sqrt(2.0 * self.inv_cutoff))

    def build(self, input_shape=None):
        init_freq = np.pi * np.arange(1, self.max_n + 1, dtype=np.float32)
        self.frequencies = self.add_weight(
            name="frequencies",
            shape=(self.max_n,),
            initializer=keras.initializers.Constant(init_freq),
            trainable=self.learnable,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, r):
        r_exp = ops.expand_dims(r, axis=-1)
        r_safe = ops.maximum(r_exp, 1e-7)
        d_scaled = r_safe * self.inv_cutoff
        return self.norm_const * ops.sin(self.frequencies * d_scaled) / r_safe


class FourierExpansion(layers.Layer):
    """Fourier expansion of scalar angular features into sine and cosine components."""

    def __init__(
        self,
        max_f: int = 4,
        interval: float = math.pi,
        scale_factor: float = 1.0,
        learnable: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_f = max_f
        self.interval = float(interval)
        self.scale_factor = float(scale_factor)
        self.learnable = learnable

    def build(self, input_shape=None):
        init_freq = np.arange(0, self.max_f + 1, dtype=np.float32)
        self.frequencies = self.add_weight(
            name="frequencies",
            shape=(self.max_f + 1,),
            initializer=keras.initializers.Constant(init_freq),
            trainable=self.learnable,
            dtype="float32",
        )
        super().build(input_shape)

    def call(self, x):
        # x: [num_items]
        # outer product with frequencies: [num_items, max_f + 1]
        x_col = ops.expand_dims(x, axis=-1)
        tmp = x_col * ops.expand_dims(self.frequencies, axis=0) * (math.pi / self.interval)
        cos_part = ops.cos(tmp)  # [N, max_f + 1]
        sin_part = ops.sin(tmp[:, 1:])  # [N, max_f]

        # Interleave cos and sin: [cos_0, sin_1, cos_1, sin_2, cos_2, ...]
        out_list = [cos_part[:, 0:1]]
        for k in range(self.max_f):
            out_list.append(sin_part[:, k:k + 1])
            out_list.append(cos_part[:, k + 1:k + 2])
        res = ops.concatenate(out_list, axis=-1)
        return (res / self.interval) * self.scale_factor


class ChebyshevRadialBasis(layers.Layer):
    """Chebyshev radial basis with polynomial cutoff envelope."""

    def __init__(
        self,
        nfunc: int = 6,
        cutoff: float = 5.0,
        cutoff_exponent: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nfunc = nfunc
        self.cutoff = float(cutoff)
        self.cutoff_exponent = cutoff_exponent

    def call(self, r):
        # Rescale r: r_tilde = 2 * (1 - |1 - r / cutoff|) - 1
        r_ratio = r / self.cutoff
        r_tilde = 2.0 * (1.0 - ops.abs(1.0 - r_ratio)) - 1.0
        r_tilde = ops.clip(r_tilde, -1.0, 1.0)

        # Evaluate Chebyshev polynomials T_1 to T_nfunc
        theta = ops.arccos(r_tilde)
        k_indices = ops.cast(ops.arange(1, self.nfunc + 1), "float32")
        # [N, 1] * [1, nfunc]
        cheb = ops.cos(ops.expand_dims(theta, axis=-1) * ops.expand_dims(k_indices, axis=0))

        env = polynomial_cutoff(r, self.cutoff, exponent=self.cutoff_exponent)
        return cheb * ops.expand_dims(env, axis=-1)


class SphericalBesselFunction(layers.Layer):
    """Spherical Bessel basis expansion j_0(k * r / cutoff)."""

    def __init__(
        self,
        max_l: int = 3,
        max_n: int = 3,
        cutoff: float = 5.0,
        smooth: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_l = max_l
        self.max_n = max_n
        self.cutoff = float(cutoff)
        self.smooth = smooth
        self.total_dim = max_n if smooth else (max_n * max_l)

    def call(self, r):
        # Standard l=0 Spherical Bessel basis:
        # sqrt(2/cutoff) * sin(n * pi * r / cutoff) / r
        n = ops.cast(ops.arange(1, self.total_dim + 1), "float32")
        r_exp = ops.expand_dims(r, axis=-1)
        r_safe = ops.maximum(r_exp, 1e-7)
        coeff = math.sqrt(2.0 / self.cutoff)
        res = coeff * ops.sin(n * math.pi / self.cutoff * r_safe) / r_safe
        if self.smooth:
            env = polynomial_cutoff(r, self.cutoff, exponent=3)
            res = res * ops.expand_dims(env, axis=-1)
        return res


class SphericalBesselWithHarmonics(layers.Layer):
    """Spherical Bessel basis combined with angular Legendre polynomials / harmonics."""

    def __init__(
        self,
        max_n: int = 3,
        max_l: int = 3,
        cutoff: float = 5.0,
        use_phi: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_n = max_n
        self.max_l = max_l
        self.cutoff = float(cutoff)
        self.use_phi = use_phi
        self.rbf = SphericalBesselFunction(max_l=max_l, max_n=max_n, cutoff=cutoff, smooth=True)

    def call(self, r, theta, phi=None):
        rbf_feat = self.rbf(r)  # [N_triples, max_n]
        # Legendre polynomials P_l(cos(theta)) for l in [0, max_l - 1]
        cos_t = ops.cos(theta)
        legendre_list = [ops.ones_like(cos_t)]
        if self.max_l > 1:
            legendre_list.append(cos_t)
        for l in range(2, self.max_l):
            p_prev = legendre_list[-1]
            p_prev2 = legendre_list[-2]
            p_curr = ((2 * l - 1) * cos_t * p_prev - (l - 1) * p_prev2) / l
            legendre_list.append(p_curr)

        sh_feat = ops.stack(legendre_list, axis=-1)  # [N_triples, max_l]
        # Outer product of radial and angular expansions
        combined = ops.expand_dims(rbf_feat, axis=-1) * ops.expand_dims(sh_feat, axis=1)
        return ops.reshape(combined, (ops.shape(r)[0], -1))


class BondExpansion(layers.Layer):
    """Radial basis function dispatcher for pair distances."""

    def __init__(
        self,
        max_l: int = 3,
        max_n: int = 3,
        cutoff: float = 5.0,
        rbf_type: Literal["SphericalBessel", "Gaussian", "ExpNorm"] = "Gaussian",
        smooth: bool = False,
        initial: float = 0.0,
        final: float = 5.0,
        num_centers: int = 100,
        width: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rbf_type = rbf_type.lower()
        self.cutoff = float(cutoff)
        self.num_centers = num_centers

        if self.rbf_type == "gaussian":
            self.rbf = GaussianExpansion(
                initial=initial,
                final=final,
                num_centers=num_centers,
                width=width,
            )
        elif self.rbf_type == "sphericalbessel":
            self.rbf = SphericalBesselFunction(
                max_l=max_l,
                max_n=max_n,
                cutoff=cutoff,
                smooth=smooth,
            )
        elif self.rbf_type == "radialbessel":
            self.rbf = RadialBesselFunction(
                max_n=max_n,
                cutoff=cutoff,
            )
        else:
            self.rbf = GaussianExpansion(
                initial=initial,
                final=final,
                num_centers=num_centers,
                width=width,
            )

    def call(self, bond_dist):
        return self.rbf(bond_dist)

