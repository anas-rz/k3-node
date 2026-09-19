from typing import Optional, List, Union
import keras
from keras import ops

from k3_node.layers.conv import ResGatedGraphConv
from k3_node.layers.pool import global_add_pool, global_max_pool, global_mean_pool


class BatchNorm1dNode(keras.layers.Layer):
    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.bn = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)

    def build(self, input_shape=None):
        self.built = True

    def call(self, x, training=False):
        return self.bn(x, training=training)


class BatchNorm1dEdge(keras.layers.Layer):
    def __init__(self, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.bn = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)

    def build(self, input_shape=None):
        self.built = True

    def call(self, edge_attr, training=False):
        return self.bn(edge_attr, training=training)


class GeneralLayer(keras.layers.Layer):
    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        has_batch_norm: bool = True,
        has_l2_norm: bool = True,
        dropout: float = 0.0,
        act: Optional[str] = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.has_l2_norm = has_l2_norm
        self.name_type = name.lower()

        if self.name_type == "linear":
            self.layer = keras.layers.Dense(out_channels, use_bias=not has_batch_norm)
        elif self.name_type == "resgatedgcnconv":
            self.layer = ResGatedGraphConv(in_channels, out_channels, bias=not has_batch_norm)
        else:
            raise ValueError(f"Unknown layer type '{name}'")

        self.has_batch_norm = has_batch_norm
        if has_batch_norm:
            self.bn = keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        else:
            self.bn = None

        self.dropout_rate = dropout
        if dropout > 0:
            self.drop = keras.layers.Dropout(dropout)
        else:
            self.drop = None

        if act is not None:
            self.act = keras.activations.get(act)
        else:
            self.act = None

    def build(self, input_shape=None):
        self.built = True

    def call(self, x, edge_index=None, training=False):
        if self.name_type == "linear":
            h = self.layer(x)
        else:
            h = self.layer(x, edge_index)

        if self.bn is not None:
            h = self.bn(h, training=training)
        if self.drop is not None:
            h = self.drop(h, training=training)
        if self.act is not None:
            h = self.act(h)
        if self.has_l2_norm:
            norm = ops.sqrt(ops.sum(ops.power(h, 2), axis=-1, keepdims=True)) + 1e-12
            h = h / norm
        return h


class GeneralMultiLayer(keras.layers.Layer):
    def __init__(
        self,
        name: str,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        num_layers: int = 1,
        has_batch_norm: bool = True,
        has_l2_norm: bool = True,
        dropout: float = 0.0,
        act: str = "relu",
        final_act: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        hidden_channels = hidden_channels or out_channels
        self.layers_list = []
        for i in range(num_layers):
            d_in = in_channels if i == 0 else hidden_channels
            d_out = out_channels if i == num_layers - 1 else hidden_channels
            act_i = None if i == num_layers - 1 and not final_act else act
            self.layers_list.append(
                GeneralLayer(
                    name=name,
                    in_channels=d_in,
                    out_channels=d_out,
                    has_batch_norm=has_batch_norm,
                    has_l2_norm=has_l2_norm,
                    dropout=dropout,
                    act=act_i,
                )
            )

    def build(self, input_shape=None):
        self.built = True

    def call(self, x, edge_index=None, training=False):
        for layer in self.layers_list:
            x = layer(x, edge_index=edge_index, training=training)
        return x


class GNNStackStage(keras.layers.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        layer_type: str = "resgatedgcnconv",
        stage_type: str = "skipsum",
        final_l2_norm: bool = True,
        has_batch_norm: bool = True,
        has_l2_norm: bool = True,
        dropout: float = 0.2,
        act: Optional[str] = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.stage_type = stage_type
        self.final_l2_norm = final_l2_norm

        self.layers_list = []
        for i in range(num_layers):
            if stage_type == "skipconcat":
                d_in = in_channels if i == 0 else in_channels + i * out_channels
            else:
                d_in = in_channels if i == 0 else out_channels
            self.layers_list.append(
                GeneralLayer(
                    name=layer_type,
                    in_channels=d_in,
                    out_channels=out_channels,
                    has_batch_norm=has_batch_norm,
                    has_l2_norm=has_l2_norm,
                    dropout=dropout,
                    act=act,
                )
            )

    def call(self, x, edge_index, training=False):
        for i, layer in enumerate(self.layers_list):
            prev_x = x
            h = layer(x, edge_index=edge_index, training=training)
            if self.stage_type == "skipsum":
                x = prev_x + h
            elif self.stage_type == "skipconcat" and i < self.num_layers - 1:
                x = ops.concatenate([prev_x, h], axis=1)
            else:
                x = h

        if self.final_l2_norm:
            norm = ops.sqrt(ops.sum(ops.power(x, 2), axis=-1, keepdims=True)) + 1e-12
            x = x / norm

        return x


class IdentityHead(keras.layers.Layer):
    def call(self, x, **kwargs):
        return x


class GNNInductiveHybridMultiHead(keras.layers.Layer):
    r"""GNN prediction head for inductive node and graph prediction tasks."""
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_node_targets: int,
        num_graph_targets: int,
        layers_post_mp: int,
        virtual_node: bool = True,
        multi_head_dim_inner: int = 32,
        graph_pooling: str = "add",
        has_bn: bool = True,
        has_l2norm: bool = True,
        dropout: float = 0.2,
        act: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.node_target_dim = num_node_targets
        self.graph_target_dim = num_graph_targets
        self.virtual_node = virtual_node
        self.graph_pooling = graph_pooling

        self.node_post_mps = [
            GeneralMultiLayer(
                name="linear",
                in_channels=dim_in,
                out_channels=1,
                hidden_channels=multi_head_dim_inner,
                num_layers=layers_post_mp,
                has_batch_norm=has_bn,
                has_l2_norm=has_l2norm,
                dropout=dropout,
                act=act,
                final_act=False,
            )
            for _ in range(num_node_targets)
        ]

        self.graph_post_mp = GeneralMultiLayer(
            name="linear",
            in_channels=dim_in,
            out_channels=num_graph_targets,
            hidden_channels=dim_in,
            num_layers=layers_post_mp,
            has_batch_norm=has_bn,
            has_l2_norm=has_l2norm,
            dropout=dropout,
            act=act,
            final_act=False,
        )

    def call(self, x, batch=None, training=False):
        node_feats = [m(x, training=training) for m in self.node_post_mps]
        node_pred = ops.concatenate(node_feats, axis=-1)

        if batch is None:
            batch = ops.zeros(ops.shape(x)[:1], dtype="int32")
        else:
            batch = ops.cast(batch, "int32")

        if self.graph_pooling == "max":
            graph_emb = global_max_pool(x, batch)
        elif self.graph_pooling == "mean":
            graph_emb = global_mean_pool(x, batch)
        else:
            graph_emb = global_add_pool(x, batch)

        graph_pred = self.graph_post_mp(graph_emb, training=training)
        return node_pred, graph_pred


class GPSE(keras.layers.Layer):
    r"""The Graph Positional and Structural Encoder (GPSE) model from the
    `"Graph Positional and Structural Encoder"
    <https://arxiv.org/abs/2307.07107>`_ paper.
    """
    def __init__(
        self,
        dim_in: int = 20,
        dim_out: int = 51,
        dim_inner: int = 512,
        layer_type: str = "resgatedgcnconv",
        layers_pre_mp: int = 1,
        layers_mp: int = 20,
        layers_post_mp: int = 2,
        num_node_targets: int = 51,
        num_graph_targets: int = 11,
        stage_type: str = "skipsum",
        has_bn: bool = True,
        head_bn: bool = False,
        final_l2norm: bool = True,
        has_l2norm: bool = True,
        dropout: float = 0.2,
        has_act: bool = True,
        final_act: bool = True,
        act: str = "relu",
        virtual_node: bool = True,
        multi_head_dim_inner: int = 32,
        graph_pooling: str = "add",
        use_repr: bool = True,
        repr_type: str = "no_post_mp",
        bernoulli_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_repr = use_repr
        self.repr_type = repr_type
        self.dim_inner = dim_inner
        self.bernoulli_threshold = bernoulli_threshold

        if layers_pre_mp > 0:
            self.pre_mp = GeneralMultiLayer(
                name="linear",
                in_channels=dim_in,
                out_channels=dim_inner,
                hidden_channels=dim_inner,
                num_layers=layers_pre_mp,
                has_batch_norm=has_bn,
                has_l2_norm=has_l2norm,
                dropout=dropout,
                act=act,
                final_act=final_act,
            )
            d_in = dim_inner
        else:
            self.pre_mp = None
            d_in = dim_in

        if layers_mp > 0:
            self.mp = GNNStackStage(
                in_channels=d_in,
                out_channels=dim_inner,
                num_layers=layers_mp,
                layer_type=layer_type,
                stage_type=stage_type,
                final_l2_norm=final_l2norm,
                has_batch_norm=has_bn,
                has_l2_norm=has_l2norm,
                dropout=dropout,
                act=act if has_act else None,
            )
        else:
            self.mp = None

        if use_repr:
            self.post_mp = IdentityHead()
        else:
            self.post_mp = GNNInductiveHybridMultiHead(
                dim_in=dim_inner,
                dim_out=dim_out,
                num_node_targets=num_node_targets,
                num_graph_targets=num_graph_targets,
                layers_post_mp=layers_post_mp,
                virtual_node=virtual_node,
                multi_head_dim_inner=multi_head_dim_inner,
                graph_pooling=graph_pooling,
                has_bn=head_bn,
                has_l2norm=has_l2norm,
                dropout=dropout,
                act=act,
            )

    def build(self, input_shape=None):
        self.built = True

    def call(self, x, edge_index=None, batch=None, training=False):
        # Support both (x, edge_index, batch) and batch object with attributes
        if hasattr(x, "x") and hasattr(x, "edge_index"):
            edge_index = x.edge_index
            batch = getattr(x, "batch", None)
            x = x.x

        if self.pre_mp is not None:
            x = self.pre_mp(x, training=training)

        if self.mp is not None:
            x = self.mp(x, edge_index, training=training)

        if self.use_repr:
            return x

        return self.post_mp(x, batch=batch, training=training)


class GPSENodeEncoder(keras.layers.Layer):
    r"""A helper linear/MLP encoder that takes the :class:`GPSE` encodings
    precomputed in the input graphs, maps them to a desired
    dimension defined by :obj:`dim_pe_out` and appends them to node features.

    Args:
        dim_emb (int): Size of final node embedding.
        dim_pe_in (int): Original dimension of GPSE encodings.
        dim_pe_out (int): Desired dimension of GPSE encodings after the encoder.
        dim_in (int, optional): Original dimension of input node features. (default: None)
        expand_x (bool, optional): Expand node features x. (default: False)
        norm_type (str, optional): Type of normalization. (default: "batchnorm")
        model_type (str, optional): Encoder model ('mlp' or 'linear'). (default: "mlp")
        n_layers (int, optional): Number of MLP layers. (default: 2)
        dropout_be (float, optional): Dropout before encoding. (default: 0.5)
        dropout_ae (float, optional): Dropout after encoding. (default: 0.2)
    """
    def __init__(
        self,
        dim_emb: int,
        dim_pe_in: int,
        dim_pe_out: int,
        dim_in: Optional[int] = None,
        expand_x: bool = False,
        norm_type: str = "batchnorm",
        model_type: str = "mlp",
        n_layers: int = 2,
        dropout_be: float = 0.5,
        dropout_ae: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert dim_emb > dim_pe_out, (
            "Desired GPSE dimension (dim_pe_out) must be smaller than "
            "the final node embedding dimension (dim_emb)."
        )

        self.expand_x = expand_x
        if expand_x:
            self.linear_x = keras.layers.Dense(dim_emb - dim_pe_out)
        else:
            self.linear_x = None

        if norm_type == "batchnorm":
            self.raw_norm = keras.layers.BatchNormalization()
        else:
            self.raw_norm = None

        self.dropout_be = keras.layers.Dropout(dropout_be)
        self.dropout_ae = keras.layers.Dropout(dropout_ae)

        if model_type == "mlp":
            layers = []
            if n_layers == 1:
                layers.append(keras.layers.Dense(dim_pe_out, activation="relu"))
            else:
                layers.append(keras.layers.Dense(2 * dim_pe_out, activation="relu"))
                for _ in range(n_layers - 2):
                    layers.append(keras.layers.Dense(2 * dim_pe_out, activation="relu"))
                layers.append(keras.layers.Dense(dim_pe_out, activation="relu"))
            self.pe_encoder = keras.Sequential(layers)
        elif model_type == "linear":
            self.pe_encoder = keras.layers.Dense(dim_pe_out)
        else:
            raise ValueError(f"Does not support '{model_type}' encoder model.")

    def call(self, x, pos_enc, training=False):
        pos_enc = self.dropout_be(pos_enc, training=training)
        if self.raw_norm is not None:
            pos_enc = self.raw_norm(pos_enc, training=training)
        pos_enc = self.pe_encoder(pos_enc)
        pos_enc = self.dropout_ae(pos_enc, training=training)

        h = self.linear_x(x) if self.expand_x else x
        return ops.concatenate([h, pos_enc], axis=1)

