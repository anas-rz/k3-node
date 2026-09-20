import math
from typing import Optional, Tuple
import numpy as np
import keras
from keras import ops

from k3_node.layers.aggr import SumAggregation
from k3_node.layers.pool import radius_graph, global_add_pool, global_mean_pool


class CosineCutoff(keras.layers.Layer):
    r"""Applies a cosine cutoff to input distances."""
    def __init__(self, cutoff: float, **kwargs):
        super().__init__(**kwargs)
        self.cutoff = cutoff

    def build(self, input_shape=None):
        self.built = True

    def call(self, distances):
        cutoffs = 0.5 * (ops.cos(distances * np.pi / self.cutoff) + 1.0)
        mask = ops.cast(distances < self.cutoff, cutoffs.dtype)
        return cutoffs * mask


class ExpNormalSmearing(keras.layers.Layer):
    r"""Applies exponential normal smearing to input distances."""
    def __init__(
        self,
        cutoff: float = 5.0,
        num_rbf: int = 128,
        trainable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        start_value = float(np.exp(-cutoff))
        means = np.linspace(start_value, 1.0, num_rbf, dtype=np.float32)
        betas = np.full((num_rbf,), (2.0 / num_rbf * (1.0 - start_value)) ** -2, dtype=np.float32)

        self.means = self.add_weight(
            name="means",
            shape=(num_rbf,),
            initializer=keras.initializers.Constant(means),
            trainable=trainable,
            dtype="float32",
        )
        self.betas = self.add_weight(
            name="betas",
            shape=(num_rbf,),
            initializer=keras.initializers.Constant(betas),
            trainable=trainable,
            dtype="float32",
        )

    def build(self, input_shape=None):
        if hasattr(self.cutoff_fn, "built") and not self.cutoff_fn.built:
            self.cutoff_fn.build(input_shape)
        self.built = True

    def call(self, dist):
        dist = ops.expand_dims(dist, -1)
        exp_dist = ops.exp(self.alpha * (-dist))
        diff = exp_dist - ops.expand_dims(self.means, 0)
        smeared = self.cutoff_fn(dist) * ops.exp(-ops.expand_dims(self.betas, 0) * ops.power(diff, 2))
        return smeared


class Sphere(keras.layers.Layer):
    r"""Computes spherical harmonics of 3D vectors up to degree lmax (1 or 2)."""
    def __init__(self, lmax: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.lmax = lmax

    def build(self, input_shape=None):
        self.built = True

    def call(self, edge_vec):
        x = edge_vec[..., 0]
        y = edge_vec[..., 1]
        z = edge_vec[..., 2]

        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        if self.lmax == 1:
            return ops.stack([sh_1_0, sh_1_1, sh_1_2], axis=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = ops.power(y, 2)
        x2z2 = ops.power(x, 2) + ops.power(z, 2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (ops.power(z, 2) - ops.power(x, 2))

        if self.lmax == 2:
            return ops.stack([
                sh_1_0, sh_1_1, sh_1_2,
                sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4,
            ], axis=-1)

        raise ValueError(f"'lmax' needs to be 1 or 2 (got {self.lmax})")


class VecLayerNorm(keras.layers.Layer):
    r"""Layer normalization for 3D vector channels."""
    def __init__(
        self,
        hidden_channels: int,
        trainable: bool = False,
        norm_type: Optional[str] = "max_min",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.norm_type = norm_type
        self.eps = 1e-12

        self.weight_param = self.add_weight(
            name="weight",
            shape=(hidden_channels,),
            initializer="ones",
            trainable=trainable,
            dtype="float32",
        )

    def build(self, input_shape=None):
        self.built = True

    def max_min_norm(self, vec):
        dist = ops.sqrt(ops.sum(ops.power(vec, 2), axis=1, keepdims=True)) + self.eps
        direct = vec / dist

        max_val = ops.max(dist, axis=-1, keepdims=True)
        min_val = ops.min(dist, axis=-1, keepdims=True)
        delta = max_val - min_val
        delta = ops.where(delta == 0, ops.ones_like(delta), delta)
        dist = (dist - min_val) / delta
        return ops.relu(dist) * direct

    def call(self, vec):
        c = ops.shape(vec)[1]
        if c == 3:
            if self.norm_type == "max_min":
                vec = self.max_min_norm(vec)
            w = ops.reshape(self.weight_param, (1, 1, self.hidden_channels))
            return vec * w
        elif c == 8:
            vec1 = vec[:, :3]
            vec2 = vec[:, 3:]
            if self.norm_type == "max_min":
                vec1 = self.max_min_norm(vec1)
                vec2 = self.max_min_norm(vec2)
            vec = ops.concatenate([vec1, vec2], axis=1)
            w = ops.reshape(self.weight_param, (1, 1, self.hidden_channels))
            return vec * w

        return vec


class Distance(keras.layers.Layer):
    r"""Pairwise distances and directions between atoms within cutoff."""
    def __init__(
        self,
        cutoff: float,
        max_num_neighbors: int = 32,
        add_self_loops: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.add_self_loops = add_self_loops

    def build(self, input_shape=None):
        self.built = True

    def call(self, pos, batch=None):
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=self.add_self_loops,
            max_num_neighbors=self.max_num_neighbors,
        )
        row = edge_index[0]
        col = edge_index[1]
        pos_row = ops.take(pos, row, axis=0)
        pos_col = ops.take(pos, col, axis=0)
        edge_vec = pos_row - pos_col
        edge_weight = ops.sqrt(ops.sum(ops.power(edge_vec, 2), axis=-1))

        if self.add_self_loops:
            mask = row != col
            edge_weight = ops.where(mask, edge_weight, ops.zeros_like(edge_weight))

        return edge_index, edge_weight, edge_vec


class NeighborEmbedding(keras.layers.Layer):
    def __init__(
        self,
        hidden_channels: int,
        num_rbf: int,
        cutoff: float,
        max_z: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.num_rbf = num_rbf
        self.embedding = keras.layers.Embedding(max_z, hidden_channels)
        self.distance_proj = keras.layers.Dense(hidden_channels)
        self.combine = keras.layers.Dense(hidden_channels)
        self.cutoff = CosineCutoff(cutoff)
        self.sum_aggr = SumAggregation()

    def build(self, input_shape=None):
        if hasattr(self.embedding, "built") and not self.embedding.built:
            self.embedding.build((None,))
        if hasattr(self.distance_proj, "built") and not self.distance_proj.built:
            self.distance_proj.build((None, self.num_rbf))
        if hasattr(self.combine, "built") and not self.combine.built:
            self.combine.build((None, self.hidden_channels * 2))
        if hasattr(self.cutoff, "built") and not self.cutoff.built:
            self.cutoff.build()
        self.built = True

    def call(self, z, x, edge_index, edge_weight, edge_attr):
        row = edge_index[0]
        col = edge_index[1]
        mask = row != col

        where_mask = ops.where(mask)
        indices = where_mask[0] if isinstance(where_mask, (tuple, list)) else where_mask
        indices = ops.reshape(indices, (-1,))

        row = ops.reshape(ops.take(row, indices, axis=0), (-1,))
        col = ops.reshape(ops.take(col, indices, axis=0), (-1,))
        edge_weight = ops.reshape(ops.take(edge_weight, indices, axis=0), (-1,))
        edge_attr = ops.take(edge_attr, indices, axis=0)

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * ops.expand_dims(C, -1)

        z = ops.cast(z, "int32")
        h = self.embedding(z)
        h_j = ops.take(h, row, axis=0)
        msg = h_j * W

        num_nodes = ops.shape(x)[0]
        x_neighbors = self.sum_aggr(msg, index=col, dim_size=num_nodes)
        x_neighbors = self.combine(ops.concatenate([x, x_neighbors], axis=1))
        return x_neighbors


class EdgeEmbedding(keras.layers.Layer):
    def __init__(self, num_rbf: int, hidden_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.num_rbf = num_rbf
        self.hidden_channels = hidden_channels
        self.edge_proj = keras.layers.Dense(hidden_channels)

    def build(self, input_shape=None):
        if hasattr(self.edge_proj, "built") and not self.edge_proj.built:
            self.edge_proj.build((None, self.num_rbf))
        self.built = True

    def call(self, edge_index, edge_attr, x):
        row = edge_index[0]
        col = edge_index[1]
        x_j = ops.take(x, row, axis=0)
        x_i = ops.take(x, col, axis=0)
        return (x_i + x_j) * self.edge_proj(edge_attr)


class ViS_MP(keras.layers.Layer):
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        last_layer: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if hidden_channels % num_heads != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = keras.layers.LayerNormalization()
        self.vec_layernorm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

        self.act = keras.activations.silu
        self.attn_activation = keras.activations.silu
        self.cutoff = CosineCutoff(cutoff)

        self.vec_proj = keras.layers.Dense(hidden_channels * 3, use_bias=False)
        self.q_proj = keras.layers.Dense(hidden_channels)
        self.k_proj = keras.layers.Dense(hidden_channels)
        self.v_proj = keras.layers.Dense(hidden_channels)
        self.dk_proj = keras.layers.Dense(hidden_channels)
        self.dv_proj = keras.layers.Dense(hidden_channels)
        self.s_proj = keras.layers.Dense(hidden_channels * 2)

        if not self.last_layer:
            self.f_proj = keras.layers.Dense(hidden_channels)
            self.w_src_proj = keras.layers.Dense(hidden_channels, use_bias=False)
            self.w_trg_proj = keras.layers.Dense(hidden_channels, use_bias=False)

        self.o_proj = keras.layers.Dense(hidden_channels * 3)
        self.sum_aggr = SumAggregation()

    def build(self, input_shape=None):
        if hasattr(self.layernorm, "built") and not self.layernorm.built:
            self.layernorm.build((None, self.hidden_channels))
        if hasattr(self.vec_layernorm, "built") and not self.vec_layernorm.built:
            self.vec_layernorm.build((None, None, self.hidden_channels))
        if hasattr(self.cutoff, "built") and not self.cutoff.built:
            self.cutoff.build()
        if hasattr(self.vec_proj, "built") and not self.vec_proj.built:
            self.vec_proj.build((None, self.hidden_channels))
        if hasattr(self.q_proj, "built") and not self.q_proj.built:
            self.q_proj.build((None, self.hidden_channels))
        if hasattr(self.k_proj, "built") and not self.k_proj.built:
            self.k_proj.build((None, self.hidden_channels))
        if hasattr(self.v_proj, "built") and not self.v_proj.built:
            self.v_proj.build((None, self.hidden_channels))
        if hasattr(self.dk_proj, "built") and not self.dk_proj.built:
            self.dk_proj.build((None, self.hidden_channels))
        if hasattr(self.dv_proj, "built") and not self.dv_proj.built:
            self.dv_proj.build((None, self.hidden_channels))
        if hasattr(self.s_proj, "built") and not self.s_proj.built:
            self.s_proj.build((None, self.hidden_channels))
        if not self.last_layer:
            if hasattr(self, "f_proj") and hasattr(self.f_proj, "built") and not self.f_proj.built:
                self.f_proj.build((None, self.hidden_channels))
            if hasattr(self, "w_src_proj") and hasattr(self.w_src_proj, "built") and not self.w_src_proj.built:
                self.w_src_proj.build((None, self.hidden_channels))
            if hasattr(self, "w_trg_proj") and hasattr(self.w_trg_proj, "built") and not self.w_trg_proj.built:
                self.w_trg_proj.build((None, self.hidden_channels))
            if hasattr(self, "t_src_proj") and hasattr(self.t_src_proj, "built") and not self.t_src_proj.built:
                self.t_src_proj.build((None, self.hidden_channels))
            if hasattr(self, "t_trg_proj") and hasattr(self.t_trg_proj, "built") and not self.t_trg_proj.built:
                self.t_trg_proj.build((None, self.hidden_channels))
        if hasattr(self.o_proj, "built") and not self.o_proj.built:
            self.o_proj.build((None, self.hidden_channels))
        self.built = True

    @staticmethod
    def vector_rejection(vec, d_ij):
        d_proj = ops.expand_dims(d_ij, 2)
        vec_proj = ops.sum(vec * d_proj, axis=1, keepdims=True)
        return vec - vec_proj * d_proj

    def call(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        row = edge_index[0]
        col = edge_index[1]
        num_nodes = ops.shape(x)[0]

        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)

        q = ops.reshape(self.q_proj(x), (-1, self.num_heads, self.head_dim))
        k = ops.reshape(self.k_proj(x), (-1, self.num_heads, self.head_dim))
        v = ops.reshape(self.v_proj(x), (-1, self.num_heads, self.head_dim))

        dk = self.act(self.dk_proj(f_ij))
        dk = ops.reshape(dk, (-1, self.num_heads, self.head_dim))
        dv = self.act(self.dv_proj(f_ij))
        dv = ops.reshape(dv, (-1, self.num_heads, self.head_dim))

        vec_proj_val = self.vec_proj(vec)
        vec1 = vec_proj_val[..., :self.hidden_channels]
        vec2 = vec_proj_val[..., self.hidden_channels:2 * self.hidden_channels]
        vec3 = vec_proj_val[..., 2 * self.hidden_channels:]
        vec_dot = ops.sum(vec1 * vec2, axis=1)

        # Message passing
        q_i = ops.take(q, col, axis=0)
        k_j = ops.take(k, row, axis=0)
        v_j = ops.take(v, row, axis=0)
        vec_j = ops.take(vec, row, axis=0)

        attn = ops.sum(q_i * k_j * dk, axis=-1)
        attn = self.attn_activation(attn) * ops.expand_dims(self.cutoff(r_ij), 1)

        v_j = v_j * dv
        v_j = ops.reshape(v_j * ops.expand_dims(attn, 2), (-1, self.hidden_channels))

        s_val = self.act(self.s_proj(v_j))
        s1 = s_val[:, :self.hidden_channels]
        s2 = s_val[:, self.hidden_channels:]

        vec_j = vec_j * ops.expand_dims(s1, 1) + ops.expand_dims(s2, 1) * ops.expand_dims(d_ij, 2)

        x_out = self.sum_aggr(v_j, index=col, dim_size=num_nodes)
        # For vec aggregation, flatten (N, channels, hid) -> sum -> reshape
        c_dim = ops.shape(vec_j)[1]
        vec_j_flat = ops.reshape(vec_j, (-1, c_dim * self.hidden_channels))
        vec_out_flat = self.sum_aggr(vec_j_flat, index=col, dim_size=num_nodes)
        vec_out = ops.reshape(vec_out_flat, (-1, c_dim, self.hidden_channels))

        o_val = self.o_proj(x_out)
        o1 = o_val[:, :self.hidden_channels]
        o2 = o_val[:, self.hidden_channels:2 * self.hidden_channels]
        o3 = o_val[:, 2 * self.hidden_channels:]

        dx = vec_dot * o2 + o3
        dvec = vec3 * ops.expand_dims(o1, 1) + vec_out

        if not self.last_layer:
            vec_i = ops.take(vec, col, axis=0)
            vec_j = ops.take(vec, row, axis=0)
            w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
            w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
            w_dot = ops.sum(w1 * w2, axis=1)
            df_ij = self.act(self.f_proj(f_ij)) * w_dot
            return dx, dvec, df_ij
        else:
            return dx, dvec, None


class ViS_MP_Vertex(ViS_MP):
    def __init__(
        self,
        num_heads: int,
        hidden_channels: int,
        cutoff: float,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        last_layer: bool = False,
        **kwargs,
    ):
        super().__init__(num_heads, hidden_channels, cutoff, vecnorm_type, trainable_vecnorm, last_layer, **kwargs)
        if not self.last_layer:
            self.f_proj = keras.layers.Dense(hidden_channels * 2)
            self.t_src_proj = keras.layers.Dense(hidden_channels, use_bias=False)
            self.t_trg_proj = keras.layers.Dense(hidden_channels, use_bias=False)

    def call(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        res = super().call(x, vec, edge_index, r_ij, f_ij, d_ij)
        if self.last_layer:
            return res
        dx, dvec, _ = res
        row = edge_index[0]
        col = edge_index[1]
        vec_i = ops.take(vec, col, axis=0)
        vec_j = ops.take(vec, row, axis=0)
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = ops.sum(w1 * w2, axis=1)

        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = ops.sum(t1 * t2, axis=1)

        f_val = self.act(self.f_proj(f_ij))
        f1 = f_val[:, :self.hidden_channels]
        f2 = f_val[:, self.hidden_channels:]
        df_ij = f1 * w_dot + f2 * t_dot
        return dx, dvec, df_ij


class GatedEquivariantBlock(keras.layers.Layer):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        scalar_activation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.intermediate_channels = intermediate_channels or hidden_channels

        self.vec1_proj = keras.layers.Dense(hidden_channels, use_bias=False)
        self.vec2_proj = keras.layers.Dense(out_channels, use_bias=False)

        self.update_net = keras.Sequential([
            keras.layers.Dense(self.intermediate_channels),
            keras.layers.Activation("silu"),
            keras.layers.Dense(out_channels * 2),
        ])
        self.scalar_activation = scalar_activation

    def build(self, input_shape=None):
        if hasattr(self.vec1_proj, "built") and not self.vec1_proj.built:
            self.vec1_proj.build((None, self.hidden_channels))
        if hasattr(self.vec2_proj, "built") and not self.vec2_proj.built:
            self.vec2_proj.build((None, self.hidden_channels))
        if hasattr(self.update_net, "built") and not self.update_net.built:
            self.update_net.build((None, self.hidden_channels * 2))
        self.built = True

    def call(self, x, v):
        vec1 = ops.sqrt(ops.sum(ops.power(self.vec1_proj(v), 2), axis=-2) + 1e-12)
        vec2 = self.vec2_proj(v)

        h = ops.concatenate([x, vec1], axis=-1)
        upd = self.update_net(h)
        x = upd[:, :self.out_channels]
        v_gate = upd[:, self.out_channels:]
        v = ops.expand_dims(v_gate, 1) * vec2

        if self.scalar_activation:
            x = ops.silu(x)

        return x, v


class EquivariantScalar(keras.layers.Layer):
    def __init__(self, hidden_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.block1 = GatedEquivariantBlock(hidden_channels, hidden_channels // 2, scalar_activation=True)
        self.block2 = GatedEquivariantBlock(hidden_channels // 2, 1, scalar_activation=False)

    def build(self, input_shape=None):
        if hasattr(self.block1, "built") and not self.block1.built:
            self.block1.build()
        if hasattr(self.block2, "built") and not self.block2.built:
            self.block2.build()
        self.built = True

    def pre_reduce(self, x, v):
        x, v = self.block1(x, v)
        x, v = self.block2(x, v)
        return x


class Atomref(keras.layers.Layer):
    def __init__(self, atomref: Optional[any] = None, max_z: int = 100, **kwargs):
        super().__init__(**kwargs)
        if atomref is None:
            atomref = np.zeros((max_z, 1), dtype=np.float32)
        else:
            atomref = np.array(atomref, dtype=np.float32)
        if atomref.ndim == 1:
            atomref = np.expand_dims(atomref, -1)

        self.atomref = keras.layers.Embedding(
            len(atomref),
            1,
            embeddings_initializer=keras.initializers.Constant(atomref),
        )

    def build(self, input_shape=None):
        if hasattr(self.atomref, "built") and not self.atomref.built:
            self.atomref.build((None,))
        self.built = True

    def call(self, x, z):
        z = ops.cast(z, "int32")
        return x + self.atomref(z)


class ViSNetBlock(keras.layers.Layer):
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lmax = lmax
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        self.distance = Distance(cutoff, max_num_neighbors=max_num_neighbors)
        self.distance_expansion = ExpNormalSmearing(cutoff, num_rbf, trainable=trainable_rbf)
        self.sphere = Sphere(lmax=lmax)
        self.neighbor_embedding = NeighborEmbedding(hidden_channels, num_rbf, cutoff, max_z=max_z)
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_channels)

        mp_class = ViS_MP_Vertex if vertex else ViS_MP
        self.vis_mp_layers = [
            mp_class(
                num_heads=num_heads,
                hidden_channels=hidden_channels,
                cutoff=cutoff,
                vecnorm_type=vecnorm_type,
                trainable_vecnorm=trainable_vecnorm,
                last_layer=(i == num_layers - 1),
            )
            for i in range(num_layers)
        ]

        self.out_norm = keras.layers.LayerNormalization()
        self.vec_out_norm = VecLayerNorm(
            hidden_channels,
            trainable=trainable_vecnorm,
            norm_type=vecnorm_type,
        )

    def build(self, input_shape=None):
        if hasattr(self.distance, "built") and not self.distance.built:
            self.distance.build((None, 3))
        if hasattr(self.distance_expansion, "built") and not self.distance_expansion.built:
            self.distance_expansion.build((None,))
        if hasattr(self.sphere, "built") and not self.sphere.built:
            self.sphere.build((None, 3))
        if hasattr(self.neighbor_embedding, "built") and not self.neighbor_embedding.built:
            self.neighbor_embedding.build((None, self.hidden_channels))
        if hasattr(self.edge_embedding, "built") and not self.edge_embedding.built:
            self.edge_embedding.build()
        for layer in self.vis_mp_layers:
            if hasattr(layer, "built") and not layer.built:
                layer.build()
        if hasattr(self.out_norm, "built") and not self.out_norm.built:
            self.out_norm.build((None, self.hidden_channels))
        if hasattr(self.vec_out_norm, "built") and not self.vec_out_norm.built:
            self.vec_out_norm.build((None, None, self.hidden_channels))
        self.built = True

    def call(self, z, pos, batch=None):
        edge_index, edge_weight, edge_vec = self.distance(pos, batch=batch)
        edge_attr = self.distance_expansion(edge_weight)
        edge_vec = self.sphere(edge_vec)

        x = ops.zeros((ops.shape(z)[0], self.hidden_channels), dtype="float32")
        x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        num_vec_channels = ((self.lmax + 1) ** 2) - 1
        vec = ops.zeros((ops.shape(z)[0], num_vec_channels, self.hidden_channels), dtype="float32")
        edge_attr = self.edge_embedding(edge_index, edge_attr, x)

        for attn in self.vis_mp_layers[:-1]:
            dx, dvec, dedge_attr = attn(x, vec, edge_index, edge_weight, edge_attr, edge_vec)
            x = x + dx
            vec = vec + dvec
            edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec)
        x = x + dx
        vec = vec + dvec

        x = self.out_norm(x)
        vec = self.vec_out_norm(vec)
        return x, vec


class ViSNet(keras.layers.Layer):
    r"""The equivariant vector-scalar interactive graph neural network (ViSNet)
    from the `"Enhancing Geometric Representations for Molecules with Equivariant
    Vector-Scalar Interactive Message Passing" <https://arxiv.org/abs/2210.16518>`_ paper.
    """
    def __init__(
        self,
        lmax: int = 1,
        vecnorm_type: Optional[str] = None,
        trainable_vecnorm: bool = False,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 128,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        max_num_neighbors: int = 32,
        vertex: bool = False,
        atomref: Optional[any] = None,
        reduce_op: str = "sum",
        mean: float = 0.0,
        std: float = 1.0,
        derivative: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.representation_model = ViSNetBlock(
            lmax=lmax,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=trainable_vecnorm,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
            vertex=vertex,
        )
        self.output_model = EquivariantScalar(hidden_channels=hidden_channels)
        self.prior_model = Atomref(atomref=atomref, max_z=max_z) if atomref is not None else None
        self.reduce_op = reduce_op
        self.mean = mean
        self.std = std
        self.derivative = derivative

    def build(self, input_shape=None):
        if hasattr(self.representation_model, "built") and not self.representation_model.built:
            self.representation_model.build(input_shape)
        if hasattr(self.output_model, "built") and not self.output_model.built:
            self.output_model.build()
        if self.prior_model is not None and hasattr(self.prior_model, "built") and not self.prior_model.built:
            self.prior_model.build()
        self.built = True

    def call(self, z, pos, batch=None):
        if batch is None:
            batch = ops.zeros(ops.shape(z), dtype="int32")
        else:
            batch = ops.cast(batch, "int32")

        x, v = self.representation_model(z, pos, batch=batch)
        x = self.output_model.pre_reduce(x, v)
        x = x * self.std

        if self.prior_model is not None:
            x = self.prior_model(x, z)

        if self.reduce_op == "mean":
            y = global_mean_pool(x, batch)
        else:
            y = global_add_pool(x, batch)

        y = y + self.mean
        return y, None
