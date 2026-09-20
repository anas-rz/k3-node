import math
import os
from typing import Optional, Union, Tuple, List, Dict, Any

import keras
from keras import layers, ops


class GaussianLayer(layers.Layer):
    r"""Gaussian basis function expansion over pairwise distances modulated by edge types.

    Args:
        num_kernel (int, optional): Number of Gaussian basis kernels. (default: ``128``)
        edge_types (int, optional): Number of pairwise edge types. (default: ``4096``)
        **kwargs: Additional layer arguments.
    """

    def __init__(self, num_kernel: int = 128, edge_types: int = 4096, **kwargs):
        super().__init__(**kwargs)
        self.num_kernel = num_kernel
        self.edge_types = edge_types

        self.means = layers.Embedding(
            input_dim=1,
            output_dim=num_kernel,
            name="means",
        )
        self.stds = layers.Embedding(
            input_dim=1,
            output_dim=num_kernel,
            name="stds",
        )
        self.mul = layers.Embedding(
            input_dim=edge_types,
            output_dim=1,
            name="mul",
        )
        self.bias = layers.Embedding(
            input_dim=edge_types,
            output_dim=1,
            name="bias",
        )

    def build(self, input_shape=None):
        if not self.built:
            self.means.build(None)
            self.stds.build(None)
            self.mul.build(None)
            self.bias.build(None)
        super().build(input_shape)

    def call(self, dist, edge_types):
        r"""
        Args:
            dist (Tensor): Pairwise distance matrix of shape ``[batch_size, num_nodes, num_nodes]``.
            edge_types (Tensor): Pairwise edge type indices of shape ``[batch_size, num_nodes, num_nodes]``.

        Returns:
            Tensor: Gaussian basis expansion of shape ``[batch_size, num_nodes, num_nodes, num_kernel]``.
        """
        mul = self.mul(edge_types)  # [B, N, N, 1]
        bias = self.bias(edge_types)  # [B, N, N, 1]
        x = mul * ops.expand_dims(dist, axis=-1) + bias  # [B, N, N, 1]

        means = ops.reshape(self.means(ops.zeros((1,), dtype="int32")), (-1,))  # [K]
        stds = ops.abs(ops.reshape(self.stds(ops.zeros((1,), dtype="int32")), (-1,))) + 1e-5  # [K]

        a = math.sqrt(2 * math.pi)
        diff = (x - means) / stds
        return ops.exp(-0.5 * ops.power(diff, 2)) / (a * stds)


class RBF(layers.Layer):
    r"""Radial Basis Function expansion over pairwise distances modulated by edge types.

    Args:
        num_kernel (int): Number of radial basis kernels.
        edge_types (int): Number of edge types.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, num_kernel: int = 128, edge_types: int = 4096, **kwargs):
        super().__init__(**kwargs)
        self.num_kernel = num_kernel
        self.edge_types = edge_types

        self.mul = layers.Embedding(input_dim=edge_types, output_dim=1, name="mul")
        self.bias = layers.Embedding(input_dim=edge_types, output_dim=1, name="bias")

    def build(self, input_shape=None):
        if not self.built:
            self.means = self.add_weight(
                shape=(self.num_kernel,),
                initializer="uniform",
                trainable=True,
                name="means",
            )
            self.temps = self.add_weight(
                shape=(self.num_kernel,),
                initializer="uniform",
                trainable=True,
                name="temps",
            )
        super().build(input_shape)

    def call(self, dist, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * ops.expand_dims(dist, axis=-1) + bias
        means = self.means
        temps = ops.abs(self.temps)
        return ops.exp(-temps * ops.power(x - means, 2))


class NonLinear(layers.Layer):
    r"""Two-layer MLP with GELU activation.

    Args:
        hidden_dim (int): Intermediate hidden dimension.
        output_dim (int): Output dimension.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, hidden_dim: int, output_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layer1 = layers.Dense(hidden_dim, name="layer1")
        self.layer2 = layers.Dense(output_dim, name="layer2")

    def build(self, input_shape=None):
        if not self.built:
            self.layer1.build((None, self.hidden_dim))
            self.layer2.build((None, self.hidden_dim))
        super().build(input_shape)

    def call(self, x):
        return self.layer2(ops.gelu(self.layer1(x)))


class SelfMultiheadAttention(layers.Layer):
    r"""Fused query-key-value self-attention with additive attention bias for 3D Graphormer.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        dropout (float, optional): Attention dropout. (default: ``0.0``)
        bias (bool, optional): Whether to use projection biases. (default: ``True``)
        scaling_factor (float, optional): Attention scaling factor. (default: ``1.0``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        scaling_factor: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_rate = dropout
        self.scaling = (self.head_dim * scaling_factor) ** -0.5

        self.in_proj = layers.Dense(embed_dim * 3, use_bias=bias, name="in_proj")
        self.out_proj = layers.Dense(embed_dim, use_bias=bias, name="out_proj")
        self.dropout = layers.Dropout(dropout)

    def build(self, input_shape=None):
        if not self.built:
            self.in_proj.build((None, None, self.embed_dim))
            self.out_proj.build((None, None, self.embed_dim))
        super().build(input_shape)

    def call(self, x, attn_bias=None, training: bool = False):
        r"""
        Args:
            x (Tensor): Node sequence tensor of shape ``[batch_size, num_nodes, embed_dim]``.
            attn_bias (Tensor, optional): Bias of shape ``[batch_size * num_heads, num_nodes, num_nodes]``.
            training (bool, optional): Training flag. (default: ``False``)

        Returns:
            Tensor: Attention output of shape ``[batch_size, num_nodes, embed_dim]``.
        """
        shape = ops.shape(x)
        batch_size, num_nodes = shape[0], shape[1]

        qkv = self.in_proj(x)
        # Split into q, k, v each of shape [batch_size, num_nodes, embed_dim]
        q = qkv[:, :, : self.embed_dim]
        k = qkv[:, :, self.embed_dim : 2 * self.embed_dim]
        v = qkv[:, :, 2 * self.embed_dim :]

        # Reshape to [batch_size * num_heads, num_nodes, head_dim]
        q = ops.reshape(
            ops.transpose(
                ops.reshape(q, (batch_size, num_nodes, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (batch_size * self.num_heads, num_nodes, self.head_dim),
        ) * self.scaling
        k = ops.reshape(
            ops.transpose(
                ops.reshape(k, (batch_size, num_nodes, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (batch_size * self.num_heads, num_nodes, self.head_dim),
        )
        v = ops.reshape(
            ops.transpose(
                ops.reshape(v, (batch_size, num_nodes, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (batch_size * self.num_heads, num_nodes, self.head_dim),
        )

        attn_weights = ops.matmul(q, ops.transpose(k, (0, 2, 1)))
        if attn_bias is not None:
            attn_weights = attn_weights + attn_bias

        attn_probs = ops.softmax(attn_weights, axis=-1)
        attn_probs = self.dropout(attn_probs, training=training)

        attn = ops.matmul(attn_probs, v)
        # Reshape back to [batch_size, num_nodes, embed_dim]
        attn = ops.reshape(
            ops.transpose(
                ops.reshape(attn, (batch_size, self.num_heads, num_nodes, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (batch_size, num_nodes, self.embed_dim),
        )
        return self.out_proj(attn)


class Graphormer3DEncoderLayer(layers.Layer):
    r"""3D Graphormer Transformer Encoder Layer with Pre-LN.

    Args:
        embedding_dim (int): Embedding dimension.
        ffn_embedding_dim (int): FFN hidden dimension.
        num_attention_heads (int): Number of attention heads.
        dropout (float, optional): Dropout probability. (default: ``0.1``)
        attention_dropout (float, optional): Attention dropout. (default: ``0.1``)
        activation_dropout (float, optional): Activation dropout. (default: ``0.1``)
        **kwargs: Additional layer arguments.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.ffn_embedding_dim = ffn_embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout_rate = dropout
        self.activation_dropout_rate = activation_dropout

        self.self_attn = SelfMultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            name="self_attn",
        )
        self.self_attn_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="self_attn_layer_norm"
        )
        self.fc1 = layers.Dense(ffn_embedding_dim, name="fc1")
        self.fc2 = layers.Dense(embedding_dim, name="fc2")
        self.final_layer_norm = layers.LayerNormalization(
            epsilon=1e-5, name="final_layer_norm"
        )

        self.dropout = layers.Dropout(dropout)
        self.act_dropout = layers.Dropout(activation_dropout)

    def build(self, input_shape=None):
        if not self.built:
            self.self_attn.build((None, None, self.embedding_dim))
            self.self_attn_layer_norm.build((None, None, self.embedding_dim))
            self.fc1.build((None, None, self.embedding_dim))
            self.fc2.build((None, None, self.ffn_embedding_dim))
            self.final_layer_norm.build((None, None, self.embedding_dim))
        super().build(input_shape)

    def call(self, x, attn_bias=None, training: bool = False):
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attn_bias=attn_bias, training=training)
        x = self.dropout(x, training=training)
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = ops.gelu(self.fc1(x))
        x = self.act_dropout(x, training=training)
        x = self.fc2(x)
        x = self.dropout(x, training=training)
        x = residual + x
        return x


class NodeTaskHead(layers.Layer):
    r"""Rotational-equivariant 3D vector force prediction head.

    Args:
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        **kwargs: Additional layer arguments.
    """

    def __init__(self, embed_dim: int, num_heads: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = layers.Dense(embed_dim, name="q_proj")
        self.k_proj = layers.Dense(embed_dim, name="k_proj")
        self.v_proj = layers.Dense(embed_dim, name="v_proj")

        self.force_proj1 = layers.Dense(1, name="force_proj1")
        self.force_proj2 = layers.Dense(1, name="force_proj2")
        self.force_proj3 = layers.Dense(1, name="force_proj3")

    def build(self, input_shape=None):
        if not self.built:
            self.q_proj.build((None, None, self.embed_dim))
            self.k_proj.build((None, None, self.embed_dim))
            self.v_proj.build((None, None, self.embed_dim))
            self.force_proj1.build((None, None, self.embed_dim))
            self.force_proj2.build((None, None, self.embed_dim))
            self.force_proj3.build((None, None, self.embed_dim))
        super().build(input_shape)

    def call(self, query, attn_bias, delta_pos, training: bool = False):
        r"""
        Args:
            query (Tensor): Node representation of shape ``[batch_size, num_nodes, embed_dim]``.
            attn_bias (Tensor): Attention bias of shape ``[batch_size * num_heads, num_nodes, num_nodes]``.
            delta_pos (Tensor): Normalized unit direction vectors of shape ``[batch_size, num_nodes, num_nodes, 3]``.
            training (bool, optional): Training flag. (default: ``False``)

        Returns:
            Tensor: Predicted 3D forces of shape ``[batch_size, num_nodes, 3]``.
        """
        shape = ops.shape(query)
        bsz, n_node = shape[0], shape[1]

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(query)
        v = self.v_proj(query)

        # Reshape to [bsz, num_heads, n_node, head_dim]
        q = ops.transpose(ops.reshape(q, (bsz, n_node, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = ops.transpose(ops.reshape(k, (bsz, n_node, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = ops.transpose(ops.reshape(v, (bsz, n_node, self.num_heads, self.head_dim)), (0, 2, 1, 3))

        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2)))  # [bsz, num_heads, n_node, n_node]
        attn_flat = ops.reshape(attn, (-1, n_node, n_node)) + attn_bias
        attn_probs = ops.softmax(attn_flat, axis=-1)
        attn_probs = ops.reshape(attn_probs, (bsz, self.num_heads, n_node, n_node))

        # rot_attn_probs: [bsz, num_heads, n_node, n_node, 3]
        rot_attn_probs = ops.expand_dims(attn_probs, axis=-1) * ops.expand_dims(delta_pos, axis=1)
        # Permute to [bsz, num_heads, 3, n_node, n_node]
        rot_attn_probs = ops.transpose(rot_attn_probs, (0, 1, 4, 2, 3))

        # Multiply with v: [bsz, num_heads, 1, n_node, head_dim]
        v_exp = ops.expand_dims(v, axis=2)
        x = ops.matmul(rot_attn_probs, v_exp)  # [bsz, num_heads, 3, n_node, head_dim]

        # Permute to [bsz, n_node, 3, num_heads, head_dim] -> [bsz, n_node, 3, embed_dim]
        x = ops.transpose(x, (0, 3, 2, 1, 4))
        x = ops.reshape(x, (bsz, n_node, 3, self.embed_dim))

        f1 = self.force_proj1(x[:, :, 0, :])  # [bsz, n_node, 1]
        f2 = self.force_proj2(x[:, :, 1, :])  # [bsz, n_node, 1]
        f3 = self.force_proj3(x[:, :, 2, :])  # [bsz, n_node, 1]

        cur_force = ops.concatenate([f1, f2, f3], axis=-1)  # [bsz, n_node, 3]
        return cur_force


class Graphormer3D(keras.Model):
    r"""Graphormer-3D model for 3D molecular structure modeling, energy, and force prediction
    from `"Benchmarking Graphormer on Large-Scale Molecular Modeling Datasets" <https://arxiv.org/abs/2203.04810>`_.

    Args:
        layers (int, optional): Number of encoder layers per block. (default: ``12``)
        blocks (int, optional): Number of repeated encoder blocks. (default: ``4``)
        embed_dim (int, optional): Hidden embedding dimension. (default: ``768``)
        ffn_embed_dim (int, optional): FFN intermediate dimension. (default: ``768``)
        attention_heads (int, optional): Number of attention heads. (default: ``48``)
        num_kernel (int, optional): Number of Gaussian basis kernels. (default: ``128``)
        atom_types (int, optional): Number of atom types. (default: ``64``)
        dropout (float, optional): Dropout probability. (default: ``0.1``)
        attention_dropout (float, optional): Attention dropout. (default: ``0.1``)
        activation_dropout (float, optional): FFN activation dropout. (default: ``0.0``)
        input_dropout (float, optional): Input features dropout. (default: ``0.0``)
        **kwargs: Additional model arguments.
    """

    def __init__(
        self,
        layers: int = 12,
        blocks: int = 4,
        embed_dim: int = 768,
        ffn_embed_dim: int = 768,
        attention_heads: int = 48,
        num_kernel: int = 128,
        atom_types: int = 64,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        input_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_encoder_layers = layers
        self.blocks = blocks
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.num_kernel = num_kernel
        self.atom_types = atom_types
        self.edge_types = atom_types * atom_types

        self.atom_encoder = keras.layers.Embedding(
            input_dim=atom_types,
            output_dim=embed_dim,
            name="atom_encoder",
        )
        self.tag_encoder = keras.layers.Embedding(
            input_dim=3,
            output_dim=embed_dim,
            name="tag_encoder",
        )
        self.input_dropout = keras.layers.Dropout(input_dropout)

        self.encoder_layers = [
            Graphormer3DEncoderLayer(
                embedding_dim=embed_dim,
                ffn_embedding_dim=ffn_embed_dim,
                num_attention_heads=attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                name=f"layers_{i}",
            )
            for i in range(layers)
        ]

        self.final_ln = keras.layers.LayerNormalization(epsilon=1e-5, name="final_ln")

        self.energy_proj = NonLinear(embed_dim, 1, name="energy_proj")
        self.energe_agg_factor = keras.layers.Embedding(
            input_dim=3, output_dim=1, name="energe_agg_factor"
        )

        self.gbf = GaussianLayer(num_kernel, self.edge_types, name="gbf")
        self.bias_proj = NonLinear(num_kernel, attention_heads, name="bias_proj")
        self.edge_proj = keras.layers.Dense(embed_dim, name="edge_proj")
        self.node_proc = NodeTaskHead(embed_dim, attention_heads, name="node_proc")

    def build(self, input_shape=None):
        if not self.built:
            self.atom_encoder.build(None)
            self.tag_encoder.build(None)
            self.gbf.build(None)
            self.bias_proj.build(None)
            self.edge_proj.build((None, self.num_kernel))
            for layer in self.encoder_layers:
                layer.build((None, None, self.embed_dim))
            self.final_ln.build((None, None, self.embed_dim))
            self.energy_proj.build(None)
            self.energe_agg_factor.build(None)
            self.node_proc.build(None)
        super().build(input_shape)

    def call(
        self,
        atoms,
        tags,
        pos,
        real_mask=None,
        training: bool = False,
    ):
        r"""Forward pass for Graphormer-3D predicting total energy and atomic forces.

        Args:
            atoms (Tensor): Atom indices of shape ``[batch_size, num_nodes]``.
            tags (Tensor): Tag indices of shape ``[batch_size, num_nodes]`` (0: fixed, 1: sub-surface, 2: surface).
            pos (Tensor): 3D atomic coordinates of shape ``[batch_size, num_nodes, 3]``.
            real_mask (Tensor, optional): Valid non-padding mask of shape ``[batch_size, num_nodes]``.
                If None, non-zero atom indices are considered valid.
            training (bool, optional): Training flag. (default: ``False``)

        Returns:
            Tuple[Tensor, Tensor]: Tuple of predicted energy ``[batch_size]`` and atomic forces
            ``[batch_size, num_nodes, 3]``.
        """
        shape = ops.shape(atoms)
        n_graph, n_node = shape[0], shape[1]

        padding_mask = ops.equal(atoms, 0)
        if real_mask is None:
            real_mask = ops.logical_not(padding_mask)

        # Pairwise displacement vectors and Euclidean distances
        delta_pos = ops.expand_dims(pos, axis=1) - ops.expand_dims(pos, axis=2)  # [B, N, N, 3]
        dist = ops.sqrt(ops.sum(ops.power(delta_pos, 2), axis=-1) + 1e-12)  # [B, N, N]
        norm_delta_pos = delta_pos / (ops.expand_dims(dist, axis=-1) + 1e-5)

        # Edge type indices: [B, N, N]
        edge_type = (
            ops.expand_dims(atoms, axis=2) * self.atom_types
            + ops.expand_dims(atoms, axis=1)
        )

        gbf_feature = self.gbf(dist, edge_type)  # [B, N, N, K]

        # Mask padding in edge features
        pad_edge_mask = ops.expand_dims(ops.expand_dims(padding_mask, axis=1), axis=-1)
        edge_features = ops.where(pad_edge_mask, 0.0, gbf_feature)

        graph_node_feature = (
            self.tag_encoder(tags)
            + self.atom_encoder(atoms)
            + self.edge_proj(ops.sum(edge_features, axis=-2))
        )

        output = self.input_dropout(graph_node_feature, training=training)

        # Attention bias: [B, N, N, num_heads] -> [B, num_heads, N, N]
        graph_attn_bias = ops.transpose(self.bias_proj(gbf_feature), (0, 3, 1, 2))
        # Mask padding: [B, 1, 1, N]
        pad_mask_bias = ops.expand_dims(ops.expand_dims(padding_mask, axis=1), axis=2)
        graph_attn_bias = ops.where(pad_mask_bias, -1e9, graph_attn_bias)
        graph_attn_bias = ops.reshape(graph_attn_bias, (-1, n_node, n_node))

        # Multi-block, multi-layer Transformer Encoder
        for _ in range(self.blocks):
            for enc_layer in self.encoder_layers:
                output = enc_layer(output, attn_bias=graph_attn_bias, training=training)

        output = self.final_ln(output)

        # Energy prediction
        eng_output = ops.squeeze(
            self.energy_proj(output) * self.energe_agg_factor(tags), axis=-1
        )
        output_mask = ops.logical_and(tags > 0, real_mask)
        eng_output = ops.sum(ops.where(output_mask, eng_output, 0.0), axis=-1)

        # Force prediction
        node_output = self.node_proc(
            output, graph_attn_bias, norm_delta_pos, training=training
        )

        return eng_output, node_output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name: str = "oc20is2re_graphormer3d_base",
        folder: str = "checkpoints",
        download: bool = True,
        **kwargs,
    ) -> "Graphormer3D":
        r"""Instantiates a Graphormer3D model with pre-trained weights."""
        cfg = get_graphormer3d_config(pretrained_name)
        cfg.update(kwargs)
        model = cls(**cfg)
        load_graphormer3d_weights(model, pretrained_name=pretrained_name, folder=folder, download=download)
        return model

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"blocks={self.blocks}, "
            f"layers={self.num_encoder_layers}, "
            f"embed_dim={self.embed_dim}, "
            f"attention_heads={self.attention_heads}, "
            f"num_kernel={self.num_kernel})"
        )


PRETRAINED_3D_URLS = {
    "oc20is2re_graphormer3d_base": "https://szheng.blob.core.windows.net/graphormer/modelzoo/oc20is2re/checkpoint_last_oc20_is2re.pt",
}


def get_graphormer3d_config(name_or_variant: str) -> Dict[str, Any]:
    r"""Returns configuration dictionary for Graphormer-3D."""
    return {
        "blocks": 4,
        "layers": 12,
        "embed_dim": 768,
        "ffn_embed_dim": 768,
        "attention_heads": 48,
        "num_kernel": 128,
        "atom_types": 64,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
    }


def download_graphormer3d_checkpoint(
    name: str = "oc20is2re_graphormer3d_base",
    folder: str = "checkpoints",
    log: bool = True,
) -> str:
    r"""Downloads a pre-trained Graphormer-3D checkpoint."""
    from k3_node.data.download import download_url

    clean_name = name.lower().replace("-", "_")
    if clean_name not in PRETRAINED_3D_URLS and name not in PRETRAINED_3D_URLS:
        raise ValueError(
            f"Unknown pretrained 3D model name '{name}'. Available: {list(PRETRAINED_3D_URLS.keys())}"
        )

    matched_key = clean_name if clean_name in PRETRAINED_3D_URLS else name
    url = PRETRAINED_3D_URLS[matched_key]
    filename = f"{matched_key}.pt"
    local_path = os.path.join(folder, filename)

    if os.path.exists(local_path):
        return local_path

    repo_alt = os.path.join("Graphormer-main", "checkpoints", filename)
    if os.path.exists(repo_alt):
        return repo_alt

    return download_url(url, folder=folder, filename=filename, log=log)


def load_graphormer3d_weights(
    model: Graphormer3D,
    checkpoint_path: Optional[str] = None,
    pretrained_name: Optional[str] = None,
    folder: str = "checkpoints",
    download: bool = True,
) -> Graphormer3D:
    r"""Loads weights from a PyTorch checkpoint into a Graphormer3D model."""
    path_to_load = checkpoint_path

    if path_to_load is None:
        if pretrained_name is None:
            raise ValueError("Either checkpoint_path or pretrained_name must be specified.")
        candidate = os.path.join(folder, f"{pretrained_name}.pt")
        if os.path.isfile(candidate):
            path_to_load = candidate
        elif download:
            path_to_load = download_graphormer3d_checkpoint(pretrained_name, folder=folder)
        else:
            raise FileNotFoundError(f"Checkpoint for '{pretrained_name}' not found at '{candidate}'.")
    elif not os.path.isfile(path_to_load) and download and pretrained_name:
        path_to_load = download_graphormer3d_checkpoint(pretrained_name, folder=folder)

    import torch
    import numpy as np

    state = torch.load(path_to_load, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict):
        state_dict = state
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(state)}")

    if not model.built:
        model.build(None)

    def _to_tensor(t):
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "numpy"):
            t = t.numpy()
        return ops.convert_to_tensor(np.array(t, dtype=np.float32), dtype="float32")

    clean_dict = {}
    for k, v in state_dict.items():
        ck = k
        if ck.startswith("encoder."):
            ck = ck[len("encoder.") :]
        clean_dict[ck] = v

    # Atom & Tag embeddings
    if "atom_encoder.weight" in clean_dict:
        model.atom_encoder.weights[0].assign(_to_tensor(clean_dict["atom_encoder.weight"]))
    if "tag_encoder.weight" in clean_dict:
        model.tag_encoder.weights[0].assign(_to_tensor(clean_dict["tag_encoder.weight"]))

    # GBF
    for name in ["means", "stds", "mul", "bias"]:
        key = f"gbf.{name}.weight"
        if key in clean_dict:
            getattr(model.gbf, name).weights[0].assign(_to_tensor(clean_dict[key]))

    # Bias projection
    for layer_name in ["layer1", "layer2"]:
        if f"bias_proj.{layer_name}.weight" in clean_dict:
            getattr(model.bias_proj, layer_name).kernel.assign(
                _to_tensor(clean_dict[f"bias_proj.{layer_name}.weight"].t())
            )
        if f"bias_proj.{layer_name}.bias" in clean_dict:
            getattr(model.bias_proj, layer_name).bias.assign(
                _to_tensor(clean_dict[f"bias_proj.{layer_name}.bias"])
            )

    # Edge projection
    if "edge_proj.weight" in clean_dict:
        model.edge_proj.kernel.assign(_to_tensor(clean_dict["edge_proj.weight"].t()))
    if "edge_proj.bias" in clean_dict:
        model.edge_proj.bias.assign(_to_tensor(clean_dict["edge_proj.bias"]))

    # Encoder layers
    for i, enc_layer in enumerate(model.encoder_layers):
        p = f"layers.{i}"
        # Self attention in_proj & out_proj
        if f"{p}.self_attn.in_proj.weight" in clean_dict:
            enc_layer.self_attn.in_proj.kernel.assign(
                _to_tensor(clean_dict[f"{p}.self_attn.in_proj.weight"].t())
            )
        if f"{p}.self_attn.in_proj.bias" in clean_dict:
            enc_layer.self_attn.in_proj.bias.assign(
                _to_tensor(clean_dict[f"{p}.self_attn.in_proj.bias"])
            )
        if f"{p}.self_attn.out_proj.weight" in clean_dict:
            enc_layer.self_attn.out_proj.kernel.assign(
                _to_tensor(clean_dict[f"{p}.self_attn.out_proj.weight"].t())
            )
        if f"{p}.self_attn.out_proj.bias" in clean_dict:
            enc_layer.self_attn.out_proj.bias.assign(
                _to_tensor(clean_dict[f"{p}.self_attn.out_proj.bias"])
            )

        # Norms
        if f"{p}.self_attn_layer_norm.weight" in clean_dict:
            enc_layer.self_attn_layer_norm.gamma.assign(
                _to_tensor(clean_dict[f"{p}.self_attn_layer_norm.weight"])
            )
        if f"{p}.self_attn_layer_norm.bias" in clean_dict:
            enc_layer.self_attn_layer_norm.beta.assign(
                _to_tensor(clean_dict[f"{p}.self_attn_layer_norm.bias"])
            )
        if f"{p}.final_layer_norm.weight" in clean_dict:
            enc_layer.final_layer_norm.gamma.assign(
                _to_tensor(clean_dict[f"{p}.final_layer_norm.weight"])
            )
        if f"{p}.final_layer_norm.bias" in clean_dict:
            enc_layer.final_layer_norm.beta.assign(
                _to_tensor(clean_dict[f"{p}.final_layer_norm.bias"])
            )

        # FFN
        if f"{p}.fc1.weight" in clean_dict:
            enc_layer.fc1.kernel.assign(_to_tensor(clean_dict[f"{p}.fc1.weight"].t()))
        if f"{p}.fc1.bias" in clean_dict:
            enc_layer.fc1.bias.assign(_to_tensor(clean_dict[f"{p}.fc1.bias"]))
        if f"{p}.fc2.weight" in clean_dict:
            enc_layer.fc2.kernel.assign(_to_tensor(clean_dict[f"{p}.fc2.weight"].t()))
        if f"{p}.fc2.bias" in clean_dict:
            enc_layer.fc2.bias.assign(_to_tensor(clean_dict[f"{p}.fc2.bias"]))

    # Final LayerNorm
    if "final_ln.weight" in clean_dict:
        model.final_ln.gamma.assign(_to_tensor(clean_dict["final_ln.weight"]))
    if "final_ln.bias" in clean_dict:
        model.final_ln.beta.assign(_to_tensor(clean_dict["final_ln.bias"]))

    # Energy head
    for layer_name in ["layer1", "layer2"]:
        if f"engergy_proj.{layer_name}.weight" in clean_dict:
            getattr(model.energy_proj, layer_name).kernel.assign(
                _to_tensor(clean_dict[f"engergy_proj.{layer_name}.weight"].t())
            )
        if f"engergy_proj.{layer_name}.bias" in clean_dict:
            getattr(model.energy_proj, layer_name).bias.assign(
                _to_tensor(clean_dict[f"engergy_proj.{layer_name}.bias"])
            )

    if "energe_agg_factor.weight" in clean_dict:
        model.energe_agg_factor.weights[0].assign(_to_tensor(clean_dict["energe_agg_factor.weight"]))

    # NodeTaskHead (Forces)
    for p_name in ["q_proj", "k_proj", "v_proj", "force_proj1", "force_proj2", "force_proj3"]:
        proj = getattr(model.node_proc, p_name)
        if f"node_proc.{p_name}.weight" in clean_dict:
            proj.kernel.assign(_to_tensor(clean_dict[f"node_proc.{p_name}.weight"].t()))
        if f"node_proc.{p_name}.bias" in clean_dict:
            proj.bias.assign(_to_tensor(clean_dict[f"node_proc.{p_name}.bias"]))

    return model
