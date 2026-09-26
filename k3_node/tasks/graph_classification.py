"""High-level Graph Classification Task."""

from typing import Any, Dict, List, Optional, Union
import keras
from keras import layers, ops
import numpy as np

from k3_node.tasks.base import BaseTask
from k3_node.tasks.backbone_resolver import resolve_backbone
from k3_node.layers import pool as k3_pool
from k3_node.loader import DataLoader


class GraphClassificationModel(keras.Model):
    r"""Internal wrapper combining a node-level GNN backbone, a global readout
    pooling operation, and a final classification dense head.
    """

    def __init__(
        self,
        backbone: keras.Model,
        pooling: str = "mean",
        hidden_channels: int = 64,
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.dropout = layers.Dropout(dropout) if dropout > 0 else None
        self.head = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        if isinstance(inputs, (tuple, list)):
            x, edge_index = inputs[0], inputs[1]
            batch = inputs[2] if len(inputs) > 2 else None
        else:
            x = inputs
            edge_index = getattr(x, "edge_index", None)
            batch = getattr(x, "batch", None)
            x = getattr(x, "x", x)

        if batch is None:
            batch = ops.zeros((ops.shape(x)[0],), dtype="int64")

        h = self.backbone((x, edge_index), training=training)

        if self.pooling in ("mean", "global_mean_pool"):
            g = k3_pool.global_mean_pool(h, batch)
        elif self.pooling in ("add", "sum", "global_add_pool"):
            g = k3_pool.global_add_pool(h, batch)
        elif self.pooling in ("max", "global_max_pool"):
            g = k3_pool.global_max_pool(h, batch)
        else:
            g = k3_pool.global_mean_pool(h, batch)

        if self.dropout is not None:
            g = self.dropout(g, training=training)
        return self.head(g)


class GraphClassifier(BaseTask):
    r"""High-level estimator for graph classification tasks (e.g., molecular property,
    bioinformatics, social graph classification).

    Args:
        backbone: GNN architecture (``"gin"``, ``"gcn"``, ``"gat"``, ``"sage"``,
            ``"pna"``, etc.) or a custom :class:`keras.Model`. (default: ``"gin"``)
        in_channels (int, optional): Size of input node features.
        hidden_channels (int, optional): Dimensionality of hidden node features. (default: ``64``)
        num_classes (int, optional): Number of graph classes.
        num_layers (int, optional): Number of GNN layers. (default: ``3``)
        pooling (str, optional): Readout pooling (``"mean"``, ``"add"``, ``"max"``). (default: ``"mean"``)
        dropout (float, optional): Dropout probability. (default: ``0.5``)
        **backbone_kwargs: Additional arguments forwarded to the backbone constructor.
    """

    def __init__(
        self,
        backbone: Union[str, keras.Model] = "gin",
        in_channels: Optional[int] = None,
        hidden_channels: int = 64,
        num_classes: Optional[int] = None,
        num_layers: int = 3,
        pooling: str = "mean",
        dropout: float = 0.5,
        **backbone_kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.pooling = pooling
        self.dropout = dropout
        self.backbone_kwargs = backbone_kwargs

    def _init_model(self, sample_data: Any, dataset: Optional[Any] = None):
        in_c = self.in_channels
        if in_c is None:
            if hasattr(sample_data, "num_node_features") and sample_data.num_node_features > 0:
                in_c = sample_data.num_node_features
            elif hasattr(sample_data, "num_features") and sample_data.num_features > 0:
                in_c = sample_data.num_features
            elif hasattr(sample_data, "x") and sample_data.x is not None:
                in_c = int(ops.shape(sample_data.x)[-1])
            else:
                raise ValueError("Could not infer in_channels.")

        out_c = self.num_classes
        if out_c is None:
            if dataset is not None and hasattr(dataset, "num_classes") and dataset.num_classes is not None:
                out_c = dataset.num_classes
            elif dataset is not None and isinstance(dataset, (list, tuple)):
                max_y = 0
                for g in dataset[:100]:
                    if hasattr(g, "y") and g.y is not None:
                        max_y = max(max_y, int(ops.convert_to_numpy(ops.max(g.y))))
                out_c = max_y + 1
            elif hasattr(sample_data, "num_classes") and sample_data.num_classes is not None and sample_data.num_classes > 1:
                out_c = sample_data.num_classes
            elif hasattr(sample_data, "y") and sample_data.y is not None:
                out_c = int(ops.convert_to_numpy(ops.max(sample_data.y))) + 1
            else:
                out_c = 2  # default binary

        out_c = max(int(out_c), 2)

        self.in_channels = in_c
        self.num_classes = out_c

        gnn = resolve_backbone(
            self.backbone,
            in_channels=in_c,
            out_channels=self.hidden_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
            **self.backbone_kwargs,
        )

        self.model = GraphClassificationModel(
            backbone=gnn,
            pooling=self.pooling,
            hidden_channels=self.hidden_channels,
            num_classes=out_c,
            dropout=self.dropout,
        )

    def fit(
        self,
        dataset: Any,
        epochs: int = 20,
        lr: float = 0.01,
        batch_size: int = 32,
        shuffle: bool = True,
        verbose: int = 1,
        callbacks: Optional[List[Any]] = None,
    ):
        r"""Trains the graph classifier."""
        if not isinstance(dataset, DataLoader):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            sample = dataset[0]
        else:
            loader = dataset
            sample = next(iter(loader))

        if self.model is None:
            self._init_model(sample, dataset=dataset)

        if not self._is_compiled:
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
            )
            self._is_compiled = True

        history = {"loss": [], "acc": []}
        for epoch in range(epochs):
            batch_losses = []
            batch_accs = []
            for batch in loader:
                x = ops.convert_to_tensor(batch.x, dtype="float32")
                edge_index = ops.convert_to_tensor(batch.edge_index, dtype="int64")
                batch_vec = ops.convert_to_tensor(batch.batch, dtype="int64")
                y = ops.convert_to_tensor(batch.y, dtype="int64")
                res = self.model.train_on_batch((x, edge_index, batch_vec), y)
                if isinstance(res, (list, tuple)):
                    batch_losses.append(float(res[0]))
                    if len(res) > 1:
                        batch_accs.append(float(res[1]))
                else:
                    batch_losses.append(float(res))

            avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            avg_acc = float(np.mean(batch_accs)) if batch_accs else 0.0
            history["loss"].append(avg_loss)
            history["acc"].append(avg_acc)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f}")

        return history

    def predict_proba(self, dataset_or_loader: Any, batch_size: int = 32):
        r"""Predicts class probabilities for graphs."""
        if not isinstance(dataset_or_loader, DataLoader):
            loader = DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False)
        else:
            loader = dataset_or_loader

        probs = []
        for batch in loader:
            x = ops.convert_to_tensor(batch.x, dtype="float32")
            edge_index = ops.convert_to_tensor(batch.edge_index, dtype="int64")
            batch_vec = ops.convert_to_tensor(batch.batch, dtype="int64")
            logits = self.model((x, edge_index, batch_vec), training=False)
            probs.append(ops.softmax(logits, axis=-1))
        return ops.concatenate(probs, axis=0)

    def predict(self, dataset_or_loader: Any, batch_size: int = 32):
        r"""Predicts discrete class labels for graphs."""
        probs = self.predict_proba(dataset_or_loader, batch_size=batch_size)
        return ops.argmax(probs, axis=-1)

    def evaluate(self, dataset_or_loader: Any, batch_size: int = 32) -> Dict[str, float]:
        r"""Evaluates classification accuracy on the dataset."""
        if not isinstance(dataset_or_loader, DataLoader):
            loader = DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False)
        else:
            loader = dataset_or_loader

        correct = 0
        total = 0
        for batch in loader:
            x = ops.convert_to_tensor(batch.x, dtype="float32")
            edge_index = ops.convert_to_tensor(batch.edge_index, dtype="int64")
            batch_vec = ops.convert_to_tensor(batch.batch, dtype="int64")
            logits = self.model((x, edge_index, batch_vec), training=False)
            pred = ops.argmax(logits, axis=-1)
            pred_np = ops.convert_to_numpy(ops.cast(pred, "int64"))
            y_np = ops.convert_to_numpy(ops.cast(batch.y, "int64"))
            correct += int((pred_np == y_np).sum())
            total += int(y_np.shape[0])

        return {"accuracy": float(correct / max(total, 1))}
