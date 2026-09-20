import os
from typing import Optional, Union, Tuple, List, Callable
import numpy as np

import keras
from keras import layers, ops

from k3_node.layers.conv.utils import softmax
from k3_node.data.download import download_google_url


def sce_loss(x, y, alpha: float = 3.0):
    r"""Scaled Cosine Error (SCE) loss from `"GraphMAE: Masked Autoencoding for Graph
    Self-Supervised Learning" <https://arxiv.org/abs/2205.10803>`_ and GraphMAE2.

    Args:
        x (Tensor): Predicted node representations.
        y (Tensor): Target node representations.
        alpha (float, optional): Scaling exponent. (default: ``3.0``)
    """
    x_norm = ops.sqrt(ops.sum(ops.power(x, 2), axis=-1, keepdims=True) + 1e-12)
    x = x / x_norm
    y_norm = ops.sqrt(ops.sum(ops.power(y, 2), axis=-1, keepdims=True) + 1e-12)
    y = y / y_norm

    cos_sim = ops.sum(x * y, axis=-1)
    diff = ops.clip(1.0 - cos_sim, 0.0, 2.0)
    loss = ops.power(diff, alpha)
    return ops.mean(loss)


def _get_activation(name: Optional[Union[str, Callable]]):
    if name is None:
        return None
    if isinstance(name, str):
        name_lower = name.lower()
        if name_lower == "prelu":
            return layers.PReLU(shared_axes=[1])
        elif name_lower == "relu":
            return layers.ReLU()
        elif name_lower == "gelu":
            return layers.Activation("gelu")
        elif name_lower == "silu":
            return layers.Activation("silu")
        elif name_lower == "elu":
            return layers.ELU()
        else:
            return layers.Activation(name)
    elif isinstance(name, layers.Layer):
        return name
    elif callable(name):
        return layers.Activation(name)
    return None


def _get_norm(name: Optional[str], dim: int):
    if name is None:
        return None
    name_lower = name.lower()
    if name_lower in ("layernorm", "layer_norm"):
        return layers.LayerNormalization(axis=-1, epsilon=1e-5)
    elif name_lower in ("batchnorm", "batch_norm"):
        return layers.BatchNormalization(axis=-1)
    return None


class GraphMAE2GATConv(layers.Layer):
    r"""GAT convolution layer matching GraphMAE2's architecture."""

    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        num_heads: int,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = False,
        activation: Optional[Union[str, Callable]] = None,
        bias: bool = True,
        norm: Optional[str] = None,
        concat_out: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.feat_drop_rate = feat_drop
        self.attn_drop_rate = attn_drop
        self.negative_slope = negative_slope
        self.use_residual = residual
        self.concat_out = concat_out
        self.use_bias = bias
        self.norm_name = norm
        self.act_name = activation

        self.fc = layers.Dense(num_heads * out_feats, use_bias=False)
        self.feat_drop = layers.Dropout(feat_drop) if feat_drop > 0.0 else None
        self.attn_drop = layers.Dropout(attn_drop) if attn_drop > 0.0 else None

        if residual and in_feats != num_heads * out_feats:
            self.res_fc = layers.Dense(num_heads * out_feats, use_bias=False)
        else:
            self.res_fc = None

        total_dim = num_heads * out_feats if concat_out else out_feats
        self.norm = _get_norm(norm, total_dim)
        self.activation = _get_activation(activation)

    def build(self, input_shape=None):
        shape = input_shape or (None, self.in_feats)
        in_dim = shape[-1] if shape is not None and shape[-1] is not None else self.in_feats

        self.fc.build((None, in_dim))
        if self.res_fc is not None:
            self.res_fc.build((None, in_dim))

        self.attn_l = self.add_weight(
            shape=(1, self.num_heads, self.out_feats),
            initializer="glorot_uniform",
            trainable=True,
            name="attn_l",
        )
        self.attn_r = self.add_weight(
            shape=(1, self.num_heads, self.out_feats),
            initializer="glorot_uniform",
            trainable=True,
            name="attn_r",
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.num_heads * self.out_feats,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
        else:
            self.bias = None

        total_dim = self.num_heads * self.out_feats if self.concat_out else self.out_feats
        if self.norm is not None:
            self.norm.build((None, total_dim))
        if self.activation is not None and hasattr(self.activation, "build"):
            self.activation.build((None, total_dim))

        self.built = True

    def call(self, x, edge_index, training=False):
        h = self.feat_drop(x, training=training) if self.feat_drop is not None else x
        feat_src = ops.reshape(self.fc(h), (-1, self.num_heads, self.out_feats))
        feat_dst = feat_src

        el = ops.sum(feat_src * self.attn_l, axis=-1, keepdims=True)
        er = ops.sum(feat_dst * self.attn_r, axis=-1, keepdims=True)

        row = ops.cast(edge_index[0], "int32")
        col = ops.cast(edge_index[1], "int32")

        el_src = ops.take(el, row, axis=0)
        er_dst = ops.take(er, col, axis=0)
        e = ops.leaky_relu(el_src + er_dst, negative_slope=self.negative_slope)

        num_nodes = ops.shape(x)[0]
        a = softmax(e, col, num_nodes=num_nodes, dim=0)
        if self.attn_drop is not None:
            a = self.attn_drop(a, training=training)

        msg = a * ops.take(feat_src, row, axis=0)
        rst = ops.segment_sum(msg, col, num_segments=num_nodes)

        if self.bias is not None:
            rst = rst + ops.reshape(self.bias, (1, self.num_heads, self.out_feats))

        if self.res_fc is not None:
            rst = rst + ops.reshape(self.res_fc(x), (num_nodes, self.num_heads, self.out_feats))

        if self.concat_out:
            rst = ops.reshape(rst, (num_nodes, self.num_heads * self.out_feats))
        else:
            rst = ops.mean(rst, axis=1)

        if self.norm is not None:
            rst = self.norm(rst)

        if self.activation is not None:
            rst = self.activation(rst)

        return rst


class GraphMAE2GAT(layers.Layer):
    r"""Multi-layer GAT encoder or decoder for GraphMAE2."""

    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        out_dim: int,
        num_layers: int,
        nhead: int,
        nhead_out: int,
        activation: Optional[str] = "prelu",
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = True,
        norm: Optional[str] = "layernorm",
        concat_out: bool = True,
        encoding: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.num_hidden = num_hidden
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.nhead_out = nhead_out
        self.concat_out = concat_out
        self.encoding = encoding

        self.gat_layers = []

        last_activation = activation if encoding else None
        last_residual = (encoding and residual)
        last_norm = norm if encoding else None

        if num_layers == 1:
            self.gat_layers.append(
                GraphMAE2GATConv(
                    in_feats=in_dim,
                    out_feats=out_dim,
                    num_heads=nhead_out,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=last_residual,
                    norm=last_norm,
                    activation=last_activation,
                    concat_out=concat_out,
                )
            )
        else:
            # Layer 0
            self.gat_layers.append(
                GraphMAE2GATConv(
                    in_feats=in_dim,
                    out_feats=num_hidden,
                    num_heads=nhead,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=residual,
                    norm=norm,
                    activation=activation,
                    concat_out=concat_out,
                )
            )
            # Intermediate layers
            for _ in range(1, num_layers - 1):
                self.gat_layers.append(
                    GraphMAE2GATConv(
                        in_feats=num_hidden * nhead,
                        out_feats=num_hidden,
                        num_heads=nhead,
                        feat_drop=feat_drop,
                        attn_drop=attn_drop,
                        negative_slope=negative_slope,
                        residual=residual,
                        norm=norm,
                        activation=activation,
                        concat_out=concat_out,
                    )
                )
            # Output layer
            self.gat_layers.append(
                GraphMAE2GATConv(
                    in_feats=num_hidden * nhead,
                    out_feats=out_dim,
                    num_heads=nhead_out,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=last_residual,
                    norm=last_norm,
                    activation=last_activation,
                    concat_out=concat_out,
                )
            )

    def build(self, input_shape=None):
        for layer in self.gat_layers:
            if hasattr(layer, "build") and not layer.built:
                layer.build()
        self.built = True

    def call(self, x, edge_index, training=False):
        h = x
        for layer in self.gat_layers:
            h = layer(h, edge_index, training=training)
        return h


class GraphMAE2(layers.Layer):
    r"""The GraphMAE2 model from `"GraphMAE2: A Decoding-Enhanced Masked
    Self-Supervised Learning Framework for Graphs" <https://arxiv.org/abs/2304.04779>`_.

    Args:
        in_dim (int): Dimensionality of input node features.
        num_hidden (int): Dimensionality of hidden node representations.
        num_layers (int, optional): Number of encoder layers. (default: ``4``)
        num_dec_layers (int, optional): Number of decoder layers. (default: ``1``)
        num_remasking (int, optional): Number of remasking views in decoder. (default: ``3``)
        nhead (int, optional): Number of attention heads in encoder. (default: ``8``)
        nhead_out (int, optional): Number of attention heads in decoder output. (default: ``1``)
        activation (str, optional): Activation function. (default: ``"prelu"``)
        feat_drop (float, optional): Node feature dropout rate. (default: ``0.2``)
        attn_drop (float, optional): Attention dropout rate. (default: ``0.1``)
        negative_slope (float, optional): LeakyReLU negative slope. (default: ``0.2``)
        residual (bool, optional): Whether to use residual connections. (default: ``True``)
        norm (str, optional): Normalization type (``"layernorm"``, ``"batchnorm"``, or ``None``). (default: ``"layernorm"``)
        mask_rate (float, optional): Fraction of input nodes to mask. (default: ``0.5``)
        remask_rate (float, optional): Fraction of latent nodes to remask. (default: ``0.5``)
        remask_method (str, optional): Remasking method (``"random"`` or ``"fixed"``). (default: ``"random"``)
        loss_fn (str, optional): Reconstruction loss type (``"sce"`` or ``"mse"``). (default: ``"sce"``)
        alpha_l (float, optional): Power exponent in Scaled Cosine Error loss. (default: ``2.0``)
        lam (float, optional): Weight of the latent prediction loss term. (default: ``1.0``)
        momentum (float, optional): Teacher EMA update momentum. (default: ``0.996``)
        delayed_ema_epoch (int, optional): Epoch to begin EMA teacher updates. (default: ``0``)
    """

    def __init__(
        self,
        in_dim: int,
        num_hidden: int,
        num_layers: int = 4,
        num_dec_layers: int = 1,
        num_remasking: int = 3,
        nhead: int = 8,
        nhead_out: int = 1,
        activation: str = "prelu",
        feat_drop: float = 0.2,
        attn_drop: float = 0.1,
        negative_slope: float = 0.2,
        residual: bool = True,
        norm: Optional[str] = "layernorm",
        mask_rate: float = 0.5,
        remask_rate: float = 0.5,
        remask_method: str = "random",
        loss_fn: str = "sce",
        alpha_l: float = 2.0,
        lam: float = 1.0,
        momentum: float = 0.996,
        delayed_ema_epoch: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_dim = in_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_dec_layers = num_dec_layers
        self.num_remasking = num_remasking
        self.nhead = nhead
        self.nhead_out = nhead_out
        self.mask_rate = mask_rate
        self.remask_rate = remask_rate
        self.remask_method = remask_method
        self.loss_fn = loss_fn
        self.alpha_l = alpha_l
        self.lam = lam
        self.momentum = momentum
        self.delayed_ema_epoch = delayed_ema_epoch

        assert num_hidden % nhead == 0, f"num_hidden ({num_hidden}) must be divisible by nhead ({nhead})"
        assert num_hidden % nhead_out == 0, f"num_hidden ({num_hidden}) must be divisible by nhead_out ({nhead_out})"

        enc_num_hidden = num_hidden // nhead
        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead

        # 1. Student Encoder
        self.encoder = GraphMAE2GAT(
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
            encoding=True,
        )

        # 2. Decoder
        self.decoder = GraphMAE2GAT(
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=num_dec_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
            encoding=False,
        )

        self.encoder_to_decoder = layers.Dense(dec_in_dim, use_bias=False)

        # 3. Projector & Predictor
        self.projector = keras.Sequential([
            layers.Dense(256),
            layers.PReLU(shared_axes=[1]),
            layers.Dense(num_hidden),
        ])

        self.predictor = keras.Sequential([
            layers.PReLU(shared_axes=[1]),
            layers.Dense(num_hidden),
        ])

        # 4. Teacher EMA networks
        self.encoder_ema = GraphMAE2GAT(
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
            encoding=True,
            trainable=False,
        )

        self.projector_ema = keras.Sequential([
            layers.Dense(256, trainable=False),
            layers.PReLU(shared_axes=[1], trainable=False),
            layers.Dense(num_hidden, trainable=False),
        ], trainable=False)

        self.enc_mask_token = self.add_weight(
            shape=(1, self.in_dim),
            initializer="glorot_normal",
            trainable=True,
            name="enc_mask_token",
        )
        self.dec_mask_token = self.add_weight(
            shape=(1, self.num_hidden),
            initializer="glorot_normal",
            trainable=True,
            name="dec_mask_token",
        )

    def build(self, input_shape=None):
        if not hasattr(self, "enc_mask_token") or self.enc_mask_token is None:
            self.enc_mask_token = self.add_weight(
                shape=(1, self.in_dim),
                initializer="glorot_normal",
                trainable=True,
                name="enc_mask_token",
            )
        if not hasattr(self, "dec_mask_token") or self.dec_mask_token is None:
            self.dec_mask_token = self.add_weight(
                shape=(1, self.num_hidden),
                initializer="glorot_normal",
                trainable=True,
                name="dec_mask_token",
            )

        self.encoder.build((None, self.in_dim))
        self.encoder_to_decoder.build((None, self.num_hidden))
        self.decoder.build((None, self.num_hidden))
        self.projector.build((None, self.num_hidden))
        self.predictor.build((None, self.num_hidden))
        self.encoder_ema.build((None, self.in_dim))
        self.projector_ema.build((None, self.num_hidden))

        # Copy initial weights from student to teacher
        for p_s, p_t in zip(self.encoder.weights, self.encoder_ema.weights):
            p_t.assign(p_s)
        for p_s, p_t in zip(self.projector.weights, self.projector_ema.weights):
            p_t.assign(p_s)

        self.built = True

    def embed(self, x, edge_index):
        r"""Generates node embeddings with the encoder."""
        if not self.built:
            self.build((None, self.in_dim))
        return self.encoder(x, edge_index)

    def encoding_mask_noise(self, x, mask_rate: Optional[float] = None, mask_nodes=None):
        r"""Masks node features for encoder input."""
        rate = self.mask_rate if mask_rate is None else mask_rate
        num_nodes = ops.shape(x)[0]

        if mask_nodes is None:
            perm = np.random.permutation(num_nodes)
            num_mask_nodes = int(rate * num_nodes)
            mask_nodes = ops.convert_to_tensor(perm[:num_mask_nodes], dtype="int32")
            keep_nodes = ops.convert_to_tensor(perm[num_mask_nodes:], dtype="int32")
        else:
            mask_nodes = ops.cast(mask_nodes, "int32")
            all_mask = np.zeros(num_nodes, dtype=bool)
            all_mask[ops.convert_to_numpy(mask_nodes)] = True
            keep_nodes = ops.convert_to_tensor(np.where(~all_mask)[0], dtype="int32")

        # Replace masked nodes with enc_mask_token
        # Create a zeroed masked version
        mask_vector = np.zeros(num_nodes, dtype=np.float32)
        mask_vector[ops.convert_to_numpy(mask_nodes)] = 1.0
        mask_tensor = ops.expand_dims(ops.convert_to_tensor(mask_vector, dtype=x.dtype), -1)

        masked_x = x * (1.0 - mask_tensor) + mask_tensor * self.enc_mask_token
        return masked_x, mask_nodes, keep_nodes

    def random_remask(self, rep, remask_rate: Optional[float] = None, remask_nodes=None):
        r"""Remasks latent representation for decoder input."""
        rate = self.remask_rate if remask_rate is None else remask_rate
        num_nodes = ops.shape(rep)[0]

        if remask_nodes is None:
            perm = np.random.permutation(num_nodes)
            num_remask_nodes = int(rate * num_nodes)
            remask_nodes = ops.convert_to_tensor(perm[:num_remask_nodes], dtype="int32")
            rekeep_nodes = ops.convert_to_tensor(perm[num_remask_nodes:], dtype="int32")
        else:
            remask_nodes = ops.cast(remask_nodes, "int32")
            all_mask = np.zeros(num_nodes, dtype=bool)
            all_mask[ops.convert_to_numpy(remask_nodes)] = True
            rekeep_nodes = ops.convert_to_tensor(np.where(~all_mask)[0], dtype="int32")

        remask_vector = np.zeros(num_nodes, dtype=np.float32)
        remask_vector[ops.convert_to_numpy(remask_nodes)] = 1.0
        remask_tensor = ops.expand_dims(ops.convert_to_tensor(remask_vector, dtype=rep.dtype), -1)

        remasked_rep = rep * (1.0 - remask_tensor) + remask_tensor * self.dec_mask_token
        return remasked_rep, remask_nodes, rekeep_nodes

    def ema_update(self, momentum: Optional[float] = None):
        r"""Updates teacher EMA parameters."""
        m = self.momentum if momentum is None else momentum
        for p_s, p_t in zip(self.encoder.weights, self.encoder_ema.weights):
            p_t.assign(p_t * m + p_s * (1.0 - m))
        for p_s, p_t in zip(self.projector.weights, self.projector_ema.weights):
            p_t.assign(p_t * m + p_s * (1.0 - m))

    def loss(
        self,
        x,
        edge_index,
        mask_nodes=None,
        targets=None,
        epoch: int = 0,
        training: bool = True,
    ):
        r"""Computes GraphMAE2 loss: attribute reconstruction loss + latent prediction loss."""
        if not self.built:
            self.build((None, self.in_dim))

        # 1. Masking
        masked_x, mask_nodes, keep_nodes = self.encoding_mask_noise(x, mask_nodes=mask_nodes)

        # 2. Student encoder
        enc_rep = self.encoder(masked_x, edge_index, training=training)

        # 3. Teacher EMA target (no gradient)
        teacher_rep = ops.stop_gradient(self.encoder_ema(x, edge_index, training=False))
        if targets is not None:
            latent_target = ops.stop_gradient(self.projector_ema(ops.take(teacher_rep, targets, axis=0)))
            latent_pred = self.predictor(self.projector(ops.take(enc_rep, targets, axis=0)))
        else:
            latent_target = ops.stop_gradient(self.projector_ema(ops.take(teacher_rep, keep_nodes, axis=0)))
            latent_pred = self.predictor(self.projector(ops.take(enc_rep, keep_nodes, axis=0)))

        loss_latent = sce_loss(latent_pred, latent_target, alpha=1.0)

        # 4. Decoder attribute reconstruction
        origin_rep = self.encoder_to_decoder(enc_rep)

        criterion = sce_loss if self.loss_fn == "sce" else (lambda pred, tgt: ops.mean(ops.power(pred - tgt, 2)))

        loss_rec_all = 0.0
        if self.remask_method == "random":
            for _ in range(self.num_remasking):
                rep, _, _ = self.random_remask(origin_rep)
                recon = self.decoder(rep, edge_index, training=training)
                x_init = ops.take(x, mask_nodes, axis=0)
                x_rec = ops.take(recon, mask_nodes, axis=0)
                loss_rec_all = loss_rec_all + criterion(x_rec, x_init, alpha=self.alpha_l) if self.loss_fn == "sce" else loss_rec_all + criterion(x_rec, x_init)
            loss_rec = loss_rec_all / float(self.num_remasking)
        else:
            # Fixed remasking
            mask_vector = np.zeros(ops.shape(x)[0], dtype=np.float32)
            mask_vector[ops.convert_to_numpy(mask_nodes)] = 1.0
            mask_tensor = ops.expand_dims(ops.convert_to_tensor(mask_vector, dtype=origin_rep.dtype), -1)
            rep = origin_rep * (1.0 - mask_tensor)
            recon = self.decoder(rep, edge_index, training=training)
            x_init = ops.take(x, mask_nodes, axis=0)
            x_rec = ops.take(recon, mask_nodes, axis=0)
            loss_rec = criterion(x_rec, x_init, alpha=self.alpha_l) if self.loss_fn == "sce" else criterion(x_rec, x_init)

        total_loss = loss_rec + self.lam * loss_latent

        if epoch >= self.delayed_ema_epoch and training:
            self.ema_update()

        return total_loss

    def call(self, x, edge_index, training: bool = False):
        r"""Forward pass: returns node embeddings by default."""
        return self.embed(x, edge_index)

    def load_weights_from_checkpoint(
        self,
        checkpoint_path: Optional[str] = None,
        dataset: Optional[str] = None,
        folder: str = "checkpoints",
        download: bool = True,
    ):
        r"""Loads weights from a PyTorch state dict checkpoint or Google Drive."""
        return load_graphmae2_weights(
            self,
            checkpoint_path=checkpoint_path,
            dataset=dataset,
            folder=folder,
            download=download,
        )

    @classmethod
    def from_pretrained(
        cls,
        dataset: str = "ogbn-arxiv",
        folder: str = "checkpoints",
        download: bool = True,
        **kwargs,
    ) -> "GraphMAE2":
        r"""Instantiates a GraphMAE2 model with pre-trained weights downloaded from Google Drive.

        Args:
            dataset (str): Dataset name (``"ogbn-arxiv"``, ``"ogbn-products"``,
                ``"mag-scholar-f"``, or ``"ogbn-papers100M"``).
            folder (str, optional): Directory to store/find checkpoints. (default: ``"checkpoints"``)
            download (bool, optional): Whether to download checkpoint if missing locally. (default: ``True``)
            **kwargs: Overrides for model hyperparameters.

        Returns:
            GraphMAE2: Model instance loaded with pre-trained weights.
        """
        key = _canonical_dataset_name(dataset)
        if key not in GRAPHMAE2_PRETRAINED:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Available pre-trained models: {list(GRAPHMAE2_PRETRAINED.keys())}"
            )
        cfg = dict(GRAPHMAE2_PRETRAINED[key])
        cfg.pop("id")
        cfg.pop("filename")
        cfg.update(kwargs)

        model = cls(**cfg)
        load_graphmae2_weights(model, dataset=key, folder=folder, download=download)
        return model

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_dim={self.in_dim}, "
            f"num_hidden={self.num_hidden}, num_layers={self.num_layers}, "
            f"num_dec_layers={self.num_dec_layers}, nhead={self.nhead})"
        )


GRAPHMAE2_PRETRAINED = {
    "ogbn-arxiv": {
        "id": "1KdU5TbAg0lQwruO7SKenoFiC2MbaZQRr",
        "filename": "gat_gat_1024_4_ogbn-arxiv_0.5_1024_checkpoint.pt",
        "in_dim": 128,
        "num_hidden": 1024,
        "num_layers": 4,
        "num_dec_layers": 1,
        "nhead": 8,
        "nhead_out": 1,
        "activation": "prelu",
        "norm": "layernorm",
        "residual": True,
    },
    "ogbn-products": {
        "id": "1Qk3bgK8H3bee3qmagH_hRJOfQmUD8PCW",
        "filename": "gat_gat_1024_4_ogbn-products_0.5_1024_checkpoint.pt",
        "in_dim": 100,
        "num_hidden": 1024,
        "num_layers": 4,
        "num_dec_layers": 1,
        "nhead": 4,
        "nhead_out": 1,
        "activation": "prelu",
        "norm": "layernorm",
        "residual": True,
    },
    "mag-scholar-f": {
        "id": "1KpQk_OKbbo4qTLQYZ84pAJDy1sh4oZv2",
        "filename": "gat_gat_1024_4_mag-scholar-f_0.5_1024_checkpoint.pt",
        "in_dim": 128,
        "num_hidden": 1024,
        "num_layers": 4,
        "num_dec_layers": 1,
        "nhead": 8,
        "nhead_out": 1,
        "activation": "prelu",
        "norm": "layernorm",
        "residual": True,
    },
    "ogbn-papers100M": {
        "id": "1zCD_vOckLfOXD1dWRY025A30QeuHsA_0",
        "filename": "gat_gat_1024_4_ogbn-papers100M_0.5_1024_checkpoint.pt",
        "in_dim": 128,
        "num_hidden": 1024,
        "num_layers": 4,
        "num_dec_layers": 1,
        "nhead": 8,
        "nhead_out": 1,
        "activation": "prelu",
        "norm": "layernorm",
        "residual": True,
    },
}

DATASET_ALIASES = {
    "arxiv": "ogbn-arxiv",
    "products": "ogbn-products",
    "mag": "mag-scholar-f",
    "mag-scholar": "mag-scholar-f",
    "papers100m": "ogbn-papers100M",
    "ogbn-papers100m": "ogbn-papers100M",
    "papers": "ogbn-papers100M",
}


def _canonical_dataset_name(name: Optional[str]) -> str:
    if not name:
        return ""
    name_clean = name.strip()
    if name_clean in GRAPHMAE2_PRETRAINED:
        return name_clean
    name_lower = name_clean.lower()
    if name_lower in DATASET_ALIASES:
        return DATASET_ALIASES[name_lower]
    for k in GRAPHMAE2_PRETRAINED:
        if k.lower() == name_lower:
            return k
    for k, v in GRAPHMAE2_PRETRAINED.items():
        if v["filename"] == name_clean:
            return k
    return name_clean


def download_graphmae2_checkpoint(
    dataset: str,
    folder: str = "checkpoints",
    log: bool = True,
) -> str:
    r"""Downloads a pre-trained GraphMAE2 checkpoint from Google Drive using download_google_url.

    Google Drive folder: https://drive.google.com/drive/folders/1GiuP0PtIZaYlJWIrjvu73ZQCJGr6kGkh

    Args:
        dataset (str): Name of dataset (``"ogbn-arxiv"``, ``"ogbn-products"``,
            ``"mag-scholar-f"``, or ``"ogbn-papers100M"``).
        folder (str, optional): Target directory to save the checkpoint. (default: ``"checkpoints"``)
        log (bool, optional): Whether to print download progress. (default: ``True``)

    Returns:
        str: Absolute path to the downloaded checkpoint file.
    """
    key = _canonical_dataset_name(dataset)
    if key not in GRAPHMAE2_PRETRAINED:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available pre-trained checkpoints: {list(GRAPHMAE2_PRETRAINED.keys())}"
        )
    info = GRAPHMAE2_PRETRAINED[key]

    target = os.path.join(folder, info["filename"])
    if os.path.exists(target):
        return target

    # Check if local file exists in GraphMAE2-main/GraphMAE2_checkpoints when using default folder
    if folder == "checkpoints":
        local_alt = os.path.join("GraphMAE2-main", "GraphMAE2_checkpoints", info["filename"])
        if os.path.exists(local_alt):
            return local_alt

    return download_google_url(
        id=info["id"],
        folder=folder,
        filename=info["filename"],
        log=log,
    )


def load_graphmae2_weights(
    model: GraphMAE2,
    checkpoint_path: Optional[str] = None,
    dataset: Optional[str] = None,
    folder: str = "checkpoints",
    download: bool = True,
):
    r"""Loads pre-trained weights from a GraphMAE2 PyTorch checkpoint (.pt).

    If the checkpoint does not exist locally and download=True, it will be automatically
    downloaded from the official Google Drive folder using download_google_url.

    Args:
        model (GraphMAE2): The target GraphMAE2 model instance.
        checkpoint_path (str, optional): Local path to .pt file or dataset name.
        dataset (str, optional): Dataset name if downloading from Google Drive.
        folder (str, optional): Directory to store downloaded checkpoints. (default: ``"checkpoints"``)
        download (bool, optional): Whether to download checkpoint if missing locally. (default: ``True``)
    """
    if checkpoint_path is None and dataset is None:
        raise ValueError("Either checkpoint_path or dataset must be specified.")

    path_to_load = checkpoint_path

    candidate_dataset = dataset or (_canonical_dataset_name(checkpoint_path) if checkpoint_path else None)
    if candidate_dataset in GRAPHMAE2_PRETRAINED:
        if checkpoint_path and os.path.isfile(checkpoint_path):
            path_to_load = checkpoint_path
        else:
            filename = GRAPHMAE2_PRETRAINED[candidate_dataset]["filename"]
            alt_local = os.path.join("GraphMAE2-main", "GraphMAE2_checkpoints", filename)
            default_local = os.path.join(folder, filename)
            if os.path.isfile(alt_local):
                path_to_load = alt_local
            elif os.path.isfile(default_local):
                path_to_load = default_local
            elif download:
                path_to_load = download_graphmae2_checkpoint(candidate_dataset, folder=folder)
            else:
                raise FileNotFoundError(f"Checkpoint for '{candidate_dataset}' not found at '{checkpoint_path}'.")
    elif checkpoint_path and not os.path.isfile(checkpoint_path):
        if download and dataset:
            path_to_load = download_graphmae2_checkpoint(dataset, folder=folder)
        else:
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")

    import torch

    state_dict = torch.load(path_to_load, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Expected a dict/state_dict in checkpoint, got {type(state_dict)}")

    if not model.built:
        model.build((None, model.in_dim))

    def _to_tensor(t):
        if hasattr(t, "detach"):
            t = t.detach()
        if hasattr(t, "numpy"):
            t = t.numpy()
        return ops.convert_to_tensor(np.array(t, dtype=np.float32), dtype="float32")

    # 1. Mask tokens
    if "enc_mask_token" in state_dict:
        model.enc_mask_token.assign(_to_tensor(state_dict["enc_mask_token"]))
    if "dec_mask_token" in state_dict:
        model.dec_mask_token.assign(_to_tensor(state_dict["dec_mask_token"]))

    # Helper for GAT module
    def _load_gat(gat_module, prefix):
        for i, layer in enumerate(gat_module.gat_layers):
            p = f"{prefix}.gat_layers.{i}"
            if f"{p}.fc.weight" in state_dict:
                layer.fc.kernel.assign(_to_tensor(state_dict[f"{p}.fc.weight"].t()))
            if f"{p}.attn_l" in state_dict:
                layer.attn_l.assign(_to_tensor(state_dict[f"{p}.attn_l"]))
            if f"{p}.attn_r" in state_dict:
                layer.attn_r.assign(_to_tensor(state_dict[f"{p}.attn_r"]))
            if f"{p}.bias" in state_dict and layer.bias is not None:
                layer.bias.assign(_to_tensor(state_dict[f"{p}.bias"]))
            if f"{p}.res_fc.weight" in state_dict and layer.res_fc is not None:
                layer.res_fc.kernel.assign(_to_tensor(state_dict[f"{p}.res_fc.weight"].t()))
            if f"{p}.activation.weight" in state_dict and hasattr(layer.activation, "alpha"):
                layer.activation.alpha.assign(_to_tensor(state_dict[f"{p}.activation.weight"]))
            if layer.norm is not None:
                if f"{p}.norm.weight" in state_dict and hasattr(layer.norm, "gamma"):
                    layer.norm.gamma.assign(_to_tensor(state_dict[f"{p}.norm.weight"]))
                if f"{p}.norm.bias" in state_dict and hasattr(layer.norm, "beta"):
                    layer.norm.beta.assign(_to_tensor(state_dict[f"{p}.norm.bias"]))

    _load_gat(model.encoder, "encoder")
    _load_gat(model.decoder, "decoder")
    _load_gat(model.encoder_ema, "encoder_ema")

    # 2. encoder_to_decoder
    if "encoder_to_decoder.weight" in state_dict:
        model.encoder_to_decoder.kernel.assign(_to_tensor(state_dict["encoder_to_decoder.weight"].t()))

    # 3. Projectors
    def _load_projector(proj_module, prefix):
        if f"{prefix}.0.weight" in state_dict:
            proj_module.layers[0].kernel.assign(_to_tensor(state_dict[f"{prefix}.0.weight"].t()))
        if f"{prefix}.0.bias" in state_dict:
            proj_module.layers[0].bias.assign(_to_tensor(state_dict[f"{prefix}.0.bias"]))
        if f"{prefix}.1.weight" in state_dict and hasattr(proj_module.layers[1], "alpha"):
            proj_module.layers[1].alpha.assign(_to_tensor(state_dict[f"{prefix}.1.weight"]))
        if f"{prefix}.2.weight" in state_dict:
            proj_module.layers[2].kernel.assign(_to_tensor(state_dict[f"{prefix}.2.weight"].t()))
        if f"{prefix}.2.bias" in state_dict:
            proj_module.layers[2].bias.assign(_to_tensor(state_dict[f"{prefix}.2.bias"]))

    _load_projector(model.projector, "projector")
    _load_projector(model.projector_ema, "projector_ema")

    # 4. Predictor
    if "predictor.0.weight" in state_dict and hasattr(model.predictor.layers[0], "alpha"):
        model.predictor.layers[0].alpha.assign(_to_tensor(state_dict["predictor.0.weight"]))
    if "predictor.1.weight" in state_dict:
        model.predictor.layers[1].kernel.assign(_to_tensor(state_dict["predictor.1.weight"].t()))
    if "predictor.1.bias" in state_dict:
        model.predictor.layers[1].bias.assign(_to_tensor(state_dict["predictor.1.bias"]))

    return model
