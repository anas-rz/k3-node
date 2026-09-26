"""High-level Link Prediction Task."""

from typing import Any, Dict, List, Optional, Tuple, Union
import keras
from keras import layers, ops
import numpy as np

from k3_node.tasks.base import BaseTask
from k3_node.tasks.backbone_resolver import resolve_backbone
from k3_node.models.utils import negative_sampling


class LinkPredictionModel(keras.Model):
    r"""Internal neural network module bundling encoder and edge decoder."""

    def __init__(
        self,
        encoder: keras.Model,
        decoder_type: str = "inner_product",
        hidden_channels: int = 64,
        **kwargs,
    ):
        kwargs.setdefault("name", "link_prediction_model")
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder_type = decoder_type.lower()
        if self.decoder_type == "mlp":
            self.decoder_mlp = keras.Sequential([
                layers.Dense(hidden_channels, activation="relu"),
                layers.Dense(1),
            ])
        else:
            self.decoder_mlp = None

    def encode(self, inputs):
        if hasattr(inputs, "inputs"):
            inputs = inputs.inputs
        return self.encoder(inputs)

    def decode(self, z, edge_label_index):
        edge_label_index = ops.convert_to_tensor(edge_label_index, dtype="int32")
        src_idx = edge_label_index[0]
        dst_idx = edge_label_index[1]
        src = ops.take(z, src_idx, axis=0)
        dst = ops.take(z, dst_idx, axis=0)

        if self.decoder_type in ("inner_product", "dot"):
            return ops.sum(src * dst, axis=-1)
        elif self.decoder_type == "cosine":
            src_norm = ops.sqrt(ops.maximum(ops.sum(ops.square(src), axis=-1, keepdims=True), 1e-8))
            dst_norm = ops.sqrt(ops.maximum(ops.sum(ops.square(dst), axis=-1, keepdims=True), 1e-8))
            return ops.sum((src / src_norm) * (dst / dst_norm), axis=-1)
        elif self.decoder_type == "mlp":
            feat = ops.concatenate([src, dst], axis=-1)
            return ops.squeeze(self.decoder_mlp(feat), axis=-1)
        else:
            raise ValueError(f"Unknown decoder type '{self.decoder_type}'. Supported: 'inner_product', 'cosine', 'mlp'.")

    def call(self, inputs, training=None):
        r"""Executes forward pass.
        inputs can be either:
          - A tuple of (graph_inputs, edge_label_index)
          - Just graph_inputs (in which case node embeddings z are returned)
        """
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2 and (isinstance(inputs[1], np.ndarray) or ops.is_tensor(inputs[1])):
            graph_inputs, edge_label_index = inputs
            z = self.encode(graph_inputs)
            return self.decode(z, edge_label_index)
        else:
            return self.encode(inputs)


class LinkPredictor(BaseTask):
    r"""High-level estimator for link prediction tasks on graphs.

    Args:
        backbone: Model architecture string (``"gcn"``, ``"gat"``, ``"sage"``,
            ``"gin"``, ``"pna"``, ``"mlp"``, etc.) or custom :class:`keras.Model`.
            (default: ``"gcn"``)
        in_channels (int, optional): Size of input node features. If not specified,
            it is automatically inferred from the dataset during :meth:`fit`.
        hidden_channels (int, optional): Dimensionality of hidden node features.
            (default: ``64``)
        out_channels (int, optional): Dimensionality of output node embeddings
            used for link scoring. (default: ``64``)
        num_layers (int, optional): Number of message passing layers. (default: ``2``)
        decoder (str, optional): Type of edge score decoder (``"inner_product"``,
            ``"cosine"``, or ``"mlp"``). (default: ``"inner_product"``)
        dropout (float, optional): Dropout probability. (default: ``0.0``)
        **backbone_kwargs: Additional arguments forwarded to the backbone constructor.
    """

    def __init__(
        self,
        backbone: Union[str, keras.Model] = "gcn",
        in_channels: Optional[int] = None,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 2,
        decoder: str = "inner_product",
        dropout: float = 0.0,
        **backbone_kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.decoder = decoder
        self.dropout = dropout
        self.backbone_kwargs = backbone_kwargs

        if isinstance(backbone, keras.Model):
            self.model = LinkPredictionModel(
                encoder=backbone,
                decoder_type=decoder,
                hidden_channels=hidden_channels,
            )

    def _init_model(self, data: Any):
        r"""Infers missing dimensions and instantiates the link prediction module."""
        in_c = self.in_channels
        if in_c is None:
            if hasattr(data, "num_node_features") and data.num_node_features > 0:
                in_c = data.num_node_features
            elif hasattr(data, "num_features") and data.num_features > 0:
                in_c = data.num_features
            elif hasattr(data, "x") and data.x is not None:
                in_c = int(ops.shape(data.x)[-1])
            else:
                raise ValueError("Could not automatically infer in_channels from data. Please specify in_channels.")

        self.in_channels = in_c

        encoder = resolve_backbone(
            self.backbone,
            in_channels=in_c,
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
            **self.backbone_kwargs,
        )

        self.model = LinkPredictionModel(
            encoder=encoder,
            decoder_type=self.decoder,
            hidden_channels=self.hidden_channels,
        )

    def fit(
        self,
        data: Any,
        edge_label_index: Optional[Any] = None,
        edge_label: Optional[Any] = None,
        epochs: int = 20,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        neg_ratio: float = 1.0,
        verbose: int = 1,
        callbacks: Optional[List[Any]] = None,
    ):
        r"""Trains the link predictor on graph connectivity."""
        if self.model is None:
            self._init_model(data)

        if not self._is_compiled:
            opt = keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
            loss = keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = [keras.metrics.BinaryAccuracy(name="acc", threshold=0.0)]
            self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
            self._is_compiled = True

        graph_inputs = self._extract_inputs(data)

        # Check for pre-split edge labels on data or passed explicitly
        has_labels = edge_label_index is not None and edge_label is not None
        if not has_labels:
            if hasattr(data, "train_edge_label_index") and hasattr(data, "train_edge_label"):
                edge_label_index = data.train_edge_label_index
                edge_label = data.train_edge_label
                has_labels = True
            elif hasattr(data, "edge_label_index") and hasattr(data, "edge_label"):
                edge_label_index = data.edge_label_index
                edge_label = data.edge_label
                has_labels = True

        num_nodes = None
        if hasattr(data, "num_nodes") and data.num_nodes is not None:
            num_nodes = data.num_nodes
        elif hasattr(data, "x") and data.x is not None:
            num_nodes = int(ops.shape(data.x)[0])

        if not has_labels:
            pos_edge_index = data.edge_index
            pos_np = ops.convert_to_numpy(pos_edge_index).astype(np.int32)
            num_pos = pos_np.shape[1]
            num_neg = int(num_pos * neg_ratio)

        history = {"loss": [], "acc": []}
        for epoch in range(epochs):
            if has_labels:
                total_edges = ops.convert_to_tensor(edge_label_index, dtype="int32")
                labels = ops.cast(ops.convert_to_tensor(edge_label), "float32")
            else:
                neg_np = negative_sampling(pos_np, num_nodes=num_nodes, num_neg_samples=num_neg)
                total_edges = ops.convert_to_tensor(np.concatenate([pos_np, neg_np], axis=1), dtype="int32")
                labels = ops.convert_to_tensor(
                    np.concatenate([np.ones(num_pos, dtype=np.float32), np.zeros(num_neg, dtype=np.float32)]),
                    dtype="float32",
                )

            if not self.model.built:
                y_pred = self.model((graph_inputs, total_edges), training=False)
                self.model.built = True
                if hasattr(self.model, "_compile_loss") and self.model._compile_loss is not None:
                    self.model._compile_loss.build(labels, y_pred)
                if hasattr(self.model, "_compile_metrics") and self.model._compile_metrics is not None:
                    self.model._compile_metrics.build(labels, y_pred)
                if self.model.optimizer is not None and not self.model.optimizer.built:
                    self.model.optimizer.build(self.model.trainable_variables)

            res = self.model.train_on_batch((graph_inputs, total_edges), labels)
            if isinstance(res, (list, tuple)):
                l, a = float(res[0]), float(res[1]) if len(res) > 1 else 0.0
            else:
                l, a = float(res), 0.0
            history["loss"].append(l)
            history["acc"].append(a)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {l:.4f} - acc: {a:.4f}")

        return history

    def encode(self, data: Any):
        r"""Computes latent node representations for the input graph."""
        if self.model is None:
            raise RuntimeError("Model is not initialized. Call fit() or construct with a model first.")
        graph_inputs = self._extract_inputs(data)
        return self.model.encode(graph_inputs)

    def predict_proba(self, data: Any, edge_label_index: Optional[Any] = None):
        r"""Predicts link existence probabilities for edge pairs."""
        if self.model is None:
            raise RuntimeError("Model is not initialized. Fit or load a model first.")

        if edge_label_index is None:
            if hasattr(data, "test_edge_label_index"):
                edge_label_index = data.test_edge_label_index
            elif hasattr(data, "edge_label_index"):
                edge_label_index = data.edge_label_index
            elif hasattr(data, "edge_index"):
                edge_label_index = data.edge_index
            else:
                raise ValueError("No edge_label_index provided and none found on data object.")

        z = self.encode(data)
        logits = self.model.decode(z, edge_label_index)
        return ops.sigmoid(logits)

    def predict(self, data: Any, edge_label_index: Optional[Any] = None, threshold: float = 0.5):
        r"""Predicts binary link presence (0 or 1) for edge pairs."""
        probs = self.predict_proba(data, edge_label_index=edge_label_index)
        return ops.cast(probs >= threshold, "int64")

    def evaluate(
        self,
        data: Any,
        edge_label_index: Optional[Any] = None,
        edge_label: Optional[Any] = None,
    ) -> Dict[str, float]:
        r"""Evaluates link prediction performance (AUC, AP, Accuracy)."""
        if edge_label_index is None and edge_label is None:
            if hasattr(data, "test_edge_label_index") and hasattr(data, "test_edge_label"):
                edge_label_index = data.test_edge_label_index
                edge_label = data.test_edge_label
            elif hasattr(data, "edge_label_index") and hasattr(data, "edge_label"):
                edge_label_index = data.edge_label_index
                edge_label = data.edge_label
            else:
                # Sample negative edges against edge_index
                pos_edges = ops.convert_to_numpy(data.edge_index).astype(np.int32)
                num_nodes = data.num_nodes if hasattr(data, "num_nodes") else int(ops.shape(data.x)[0])
                neg_edges = negative_sampling(pos_edges, num_nodes=num_nodes, num_neg_samples=pos_edges.shape[1])
                edge_label_index = np.concatenate([pos_edges, neg_edges], axis=1)
                edge_label = np.concatenate([np.ones(pos_edges.shape[1]), np.zeros(neg_edges.shape[1])])

        probs = self.predict_proba(data, edge_label_index=edge_label_index)
        probs_np = ops.convert_to_numpy(probs).flatten()
        y_np = ops.convert_to_numpy(edge_label).flatten()

        metrics = {}
        # Accuracy
        preds_bin = (probs_np >= 0.5).astype(np.float32)
        metrics["accuracy"] = float(np.mean(preds_bin == y_np))

        # AUC and AP via scikit-learn when available
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            metrics["auc"] = float(roc_auc_score(y_np, probs_np))
            metrics["ap"] = float(average_precision_score(y_np, probs_np))
        except ImportError:
            pass

        return metrics
