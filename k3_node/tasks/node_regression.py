"""High-level Node Regression Task."""

from typing import Any, Dict, List, Optional, Union
import keras
from keras import ops

from k3_node.tasks.base import BaseTask
from k3_node.tasks.backbone_resolver import resolve_backbone


class NodeRegressor(BaseTask):
    r"""High-level estimator for node regression tasks.

    Args:
        backbone: Model architecture string (``"gcn"``, ``"gat"``, ``"sage"``,
            ``"gin"``, ``"pna"``, ``"mlp"``, etc.) or a custom :class:`keras.Model`.
            (default: ``"gcn"``)
        in_channels (int, optional): Size of input node features.
        hidden_channels (int, optional): Dimensionality of hidden node features. (default: ``64``)
        out_channels (int, optional): Number of continuous target variables. (default: ``1``)
        num_layers (int, optional): Number of message passing layers. (default: ``2``)
        dropout (float, optional): Dropout probability. (default: ``0.0``)
        loss: Regression loss (``"mse"``, ``"mae"``, or a Keras loss instance). (default: ``"mse"``)
        **backbone_kwargs: Additional arguments forwarded to the backbone constructor.
    """

    def __init__(
        self,
        backbone: Union[str, keras.Model] = "gcn",
        in_channels: Optional[int] = None,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 2,
        dropout: float = 0.0,
        loss: str = "mse",
        **backbone_kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.loss_name = loss
        self.backbone_kwargs = backbone_kwargs

        if isinstance(backbone, keras.Model):
            self.model = backbone

    def _init_model(self, data: Any):
        in_c = self.in_channels
        if in_c is None:
            if hasattr(data, "num_node_features") and data.num_node_features > 0:
                in_c = data.num_node_features
            elif hasattr(data, "num_features") and data.num_features > 0:
                in_c = data.num_features
            elif hasattr(data, "x") and data.x is not None:
                in_c = int(ops.shape(data.x)[-1])
            else:
                raise ValueError("Could not automatically infer in_channels from data.")

        self.in_channels = in_c
        self.model = resolve_backbone(
            self.backbone,
            in_channels=in_c,
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
            dropout=self.dropout,
            **self.backbone_kwargs,
        )

    def fit(
        self,
        data: Any,
        epochs: int = 20,
        lr: float = 0.01,
        mask: Optional[str] = "train_mask",
        verbose: int = 1,
        callbacks: Optional[List[Any]] = None,
    ):
        r"""Trains the node regressor on the provided graph data."""
        if self.model is None:
            self._init_model(data)

        if not self._is_compiled:
            loss = keras.losses.MeanSquaredError() if self.loss_name == "mse" else keras.losses.MeanAbsoluteError()
            self.model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss=loss,
                weighted_metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
            )
            self._is_compiled = True

        inputs = self._extract_inputs(data)
        y = ops.cast(data.y, "float32")
        m = getattr(data, mask) if mask and hasattr(data, mask) else None
        sample_weight = ops.cast(m, "float32") if m is not None else None

        def gen_fn():
            while True:
                if sample_weight is not None:
                    yield inputs, y, sample_weight
                else:
                    yield inputs, y

        return self.model.fit(
            gen_fn(),
            steps_per_epoch=1,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )

    def predict(self, data: Any, mask: Optional[str] = None):
        r"""Returns continuous predictions for nodes."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        inputs = self._extract_inputs(data)
        pred = self.model(inputs, training=False)
        if mask is not None and hasattr(data, mask):
            pred = pred[getattr(data, mask)]
        return pred

    def evaluate(self, data: Any, mask: Optional[str] = "test_mask") -> Dict[str, float]:
        r"""Evaluates Mean Absolute Error and Mean Squared Error."""
        pred = self.predict(data, mask=mask)
        y = data.y
        if mask is not None and hasattr(data, mask):
            y = y[getattr(data, mask)]
        diff = ops.cast(pred, "float32") - ops.cast(y, "float32")
        mae = float(ops.convert_to_numpy(ops.mean(ops.abs(diff))))
        mse = float(ops.convert_to_numpy(ops.mean(ops.square(diff))))
        return {"mae": mae, "mse": mse, "loss": mse}
