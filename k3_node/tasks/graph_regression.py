"""High-level Graph / Molecular Regression Task."""

from typing import Any, Dict, List, Optional, Union
import keras
from keras import layers, ops
import numpy as np

from k3_node.tasks.base import BaseTask
from k3_node.tasks.backbone_resolver import resolve_backbone
from k3_node.layers import pool as k3_pool
from k3_node.loader import DataLoader


class GraphRegressionModel(keras.Model):
    r"""Combines GNN backbone, pooling, and a continuous regression head."""

    def __init__(
        self,
        backbone: keras.Model,
        pooling: str = "add",
        out_channels: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.dropout = layers.Dropout(dropout) if dropout > 0 else None
        self.head = layers.Dense(out_channels)

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

        if self.pooling in ("add", "sum", "global_add_pool"):
            g = k3_pool.global_add_pool(h, batch)
        elif self.pooling in ("mean", "global_mean_pool"):
            g = k3_pool.global_mean_pool(h, batch)
        elif self.pooling in ("max", "global_max_pool"):
            g = k3_pool.global_max_pool(h, batch)
        else:
            g = k3_pool.global_add_pool(h, batch)

        if self.dropout is not None:
            g = self.dropout(g, training=training)
        return self.head(g)


class GraphRegressor(BaseTask):
    r"""High-level estimator for graph and molecular property regression tasks.

    Args:
        backbone: Architecture (``"schnet"``, ``"dimenet++"``, ``"attentive_fp"``,
            ``"pna"``, ``"gin"``, ``"gcn"``, etc.) or custom model. (default: ``"gin"``)
        in_channels (int, optional): Size of input node features.
        hidden_channels (int, optional): Hidden feature dimension. (default: ``64``)
        out_channels (int, optional): Number of continuous target variables. (default: ``1``)
        num_layers (int, optional): Number of GNN layers. (default: ``3``)
        pooling (str, optional): Readout pooling (``"add"``, ``"mean"``, ``"max"``). (default: ``"add"``)
        loss (str, optional): Loss function (``"mae"`` or ``"mse"``). (default: ``"mae"``)
        **backbone_kwargs: Additional arguments forwarded to the backbone constructor.
    """

    def __init__(
        self,
        backbone: Union[str, keras.Model] = "gin",
        in_channels: Optional[int] = None,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 3,
        pooling: str = "add",
        loss: str = "mae",
        **backbone_kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.pooling = pooling
        self.loss_name = loss
        self.backbone_kwargs = backbone_kwargs

    def _init_model(self, sample_data: Any):
        name = str(self.backbone).lower()
        is_direct_graph_model = any(k in name for k in ("schnet", "dimenet", "attentive"))

        in_c = self.in_channels or getattr(sample_data, "num_node_features", None) or getattr(sample_data, "num_features", None) or 16
        self.in_channels = in_c

        resolved = resolve_backbone(
            self.backbone,
            in_channels=in_c,
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            **self.backbone_kwargs,
        )

        if is_direct_graph_model:
            self.model = resolved
        else:
            self.model = GraphRegressionModel(
                backbone=resolved,
                pooling=self.pooling,
                out_channels=self.out_channels,
            )

    def fit(
        self,
        dataset: Any,
        epochs: int = 20,
        lr: float = 0.001,
        batch_size: int = 32,
        shuffle: bool = True,
        verbose: int = 1,
        callbacks: Optional[List[Any]] = None,
    ):
        r"""Trains the graph regressor."""
        if not isinstance(dataset, DataLoader):
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            sample = dataset[0]
        else:
            loader = dataset
            sample = next(iter(loader))

        if self.model is None:
            self._init_model(sample)

        if not self._is_compiled:
            loss = keras.losses.MeanAbsoluteError() if self.loss_name == "mae" else keras.losses.MeanSquaredError()
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss=loss,
                metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
            )
            self._is_compiled = True

        history = {"loss": [], "mae": []}
        for epoch in range(epochs):
            batch_losses = []
            batch_maes = []
            for batch in loader:
                x = ops.convert_to_tensor(batch.x, dtype="float32")
                edge_index = ops.convert_to_tensor(batch.edge_index, dtype="int64")
                batch_vec = ops.convert_to_tensor(batch.batch, dtype="int64")
                y = ops.cast(ops.convert_to_tensor(batch.y), "float32")
                res = self.model.train_on_batch((x, edge_index, batch_vec), y)
                if isinstance(res, (list, tuple)):
                    batch_losses.append(float(res[0]))
                    if len(res) > 1:
                        batch_maes.append(float(res[1]))
                else:
                    batch_losses.append(float(res))

            avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            avg_mae = float(np.mean(batch_maes)) if batch_maes else 0.0
            history["loss"].append(avg_loss)
            history["mae"].append(avg_mae)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f} - mae: {avg_mae:.4f}")

        return history

    def predict(self, dataset_or_loader: Any, batch_size: int = 32):
        r"""Returns continuous predictions for graphs."""
        if not isinstance(dataset_or_loader, DataLoader):
            loader = DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False)
        else:
            loader = dataset_or_loader

        preds = []
        for batch in loader:
            x = ops.convert_to_tensor(batch.x, dtype="float32")
            edge_index = ops.convert_to_tensor(batch.edge_index, dtype="int64")
            batch_vec = ops.convert_to_tensor(batch.batch, dtype="int64")
            pred = self.model((x, edge_index, batch_vec), training=False)
            preds.append(pred)
        return ops.concatenate(preds, axis=0)

    def evaluate(self, dataset_or_loader: Any, batch_size: int = 32) -> Dict[str, float]:
        r"""Evaluates MAE on the dataset."""
        preds = self.predict(dataset_or_loader, batch_size=batch_size)
        ys = []
        loader = dataset_or_loader if isinstance(dataset_or_loader, DataLoader) else DataLoader(dataset_or_loader, batch_size=batch_size, shuffle=False)
        for batch in loader:
            ys.append(ops.cast(batch.y, "float32"))
        y_all = ops.concatenate(ys, axis=0)
        diff = ops.abs(preds - y_all)
        mae = float(ops.convert_to_numpy(ops.mean(diff)))
        mse = float(ops.convert_to_numpy(ops.mean(ops.square(diff))))
        return {"mae": mae, "mse": mse, "loss": mae}
