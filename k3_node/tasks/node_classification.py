"""High-level Node Classification Task."""

from typing import Any, Dict, List, Optional, Union
import keras
from keras import ops

from k3_node.tasks.base import BaseTask
from k3_node.tasks.backbone_resolver import resolve_backbone


class NodeClassifier(BaseTask):
    r"""High-level estimator for node classification tasks.

    Args:
        backbone: Model architecture string (``"gcn"``, ``"gat"``, ``"sage"``,
            ``"gin"``, ``"pna"``, ``"mlp"``, etc.) or a custom :class:`keras.Model`.
            (default: ``"gcn"``)
        in_channels (int, optional): Size of input node features. If not specified,
            it is automatically inferred from the dataset during :meth:`fit`.
        hidden_channels (int, optional): Dimensionality of hidden node features.
            (default: ``64``)
        out_channels (int, optional): Number of target classes. If not specified,
            it is automatically inferred from the dataset during :meth:`fit`.
        num_layers (int, optional): Number of message passing layers. (default: ``2``)
        dropout (float, optional): Dropout probability. (default: ``0.5``)
        multi_label (bool, optional): If :obj:`True`, treats the problem as multi-label
            binary classification using binary crossentropy. (default: ``False``)
        **backbone_kwargs: Additional arguments forwarded to the backbone constructor.
    """

    def __init__(
        self,
        backbone: Union[str, keras.Model] = "gcn",
        in_channels: Optional[int] = None,
        hidden_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.5,
        multi_label: bool = False,
        **backbone_kwargs,
    ):
        super().__init__()
        self.backbone = backbone
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.multi_label = multi_label
        self.backbone_kwargs = backbone_kwargs

        if isinstance(backbone, keras.Model):
            self.model = backbone

    def _init_model(self, data: Any):
        r"""Infers missing dimensions and instantiates the backbone model."""
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

        out_c = self.out_channels
        if out_c is None:
            if hasattr(data, "num_classes") and data.num_classes is not None:
                out_c = data.num_classes
            elif hasattr(data, "y") and data.y is not None:
                y = data.y
                if self.multi_label:
                    out_c = int(ops.shape(y)[-1])
                else:
                    out_c = int(ops.convert_to_numpy(ops.max(y))) + 1
            else:
                raise ValueError("Could not automatically infer out_channels from data. Please specify out_channels.")

        if not self.multi_label:
            out_c = max(int(out_c), 2)

        self.in_channels = in_c
        self.out_channels = out_c

        self.model = resolve_backbone(
            self.backbone,
            in_channels=in_c,
            out_channels=out_c,
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
        weight_decay: float = 5e-4,
        mask: Optional[str] = "train_mask",
        val_mask: Optional[str] = "val_mask",
        verbose: int = 1,
        callbacks: Optional[List[Any]] = None,
    ):
        r"""Trains the node classifier on the provided graph data."""
        if self.model is None:
            self._init_model(data)

        if not self._is_compiled:
            opt = keras.optimizers.Adam(learning_rate=lr, weight_decay=weight_decay)
            if self.multi_label:
                loss = keras.losses.BinaryCrossentropy(from_logits=True)
                metrics = [keras.metrics.BinaryAccuracy(name="acc")]
            else:
                loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
                metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")]

            self.model.compile(
                optimizer=opt,
                loss=loss,
                weighted_metrics=metrics,
            )
            self._is_compiled = True

        # Generate training batches
        if hasattr(data, "to_generator"):
            gen = data.to_generator(mask=mask)
        else:
            inputs = self._extract_inputs(data)
            y = ops.convert_to_tensor(data.y)
            m = getattr(data, mask) if mask and hasattr(data, mask) else None
            sample_weight = ops.cast(m, "float32") if m is not None else None

            def gen_fn():
                while True:
                    if sample_weight is not None:
                        yield inputs, y, sample_weight
                    else:
                        yield inputs, y

            gen = gen_fn()

        return self.model.fit(
            gen,
            steps_per_epoch=1,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
        )

    def predict_proba(self, data: Any, mask: Optional[str] = None):
        r"""Returns class probabilities for nodes."""
        if self.model is None:
            raise RuntimeError("Model is not initialized. Fit or load a model first.")

        inputs = self._extract_inputs(data)
        logits = self.model(inputs, training=False)

        if self.multi_label:
            probs = ops.sigmoid(logits)
        else:
            probs = ops.softmax(logits, axis=-1)

        if mask is not None and hasattr(data, mask):
            m = getattr(data, mask)
            probs = probs[m]
        return probs

    def predict(self, data: Any, mask: Optional[str] = None):
        r"""Predicts discrete class labels for nodes."""
        probs = self.predict_proba(data, mask=mask)
        if self.multi_label:
            return ops.cast(probs > 0.5, "int64")
        return ops.argmax(probs, axis=-1)

    def evaluate(self, data: Any, mask: Optional[str] = "test_mask") -> Dict[str, float]:
        r"""Evaluates classification accuracy on a given mask."""
        pred = self.predict(data, mask=mask)
        y = data.y
        if mask is not None and hasattr(data, mask):
            m = getattr(data, mask)
            y = y[m]

        y_cast = ops.cast(y, "int64")
        pred_cast = ops.cast(pred, "int64")
        acc = float(ops.convert_to_numpy(ops.mean(ops.cast(pred_cast == y_cast, "float32"))))
        return {"accuracy": acc}
