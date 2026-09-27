"""Metrics for graph learning tasks."""
import keras
from keras import ops


@keras.saving.register_keras_serializable(package="k3_node")
class F1Score(keras.metrics.F1Score):
    r"""F1 score that also accepts logits, e.g. for multi-label node classification.

    Same as :class:`keras.metrics.F1Score`, plus ``from_logits``: when :obj:`True`, a sigmoid is
    applied to the predictions first, so models can output logits (as used with
    ``BinaryCrossentropy(from_logits=True)``).

    Example:
        ```python
        import numpy as np
        from k3_node.metrics import F1Score

        f1 = F1Score(average="micro", from_logits=True)
        y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype="float32")
        logits = np.array([[2.0, -1.0, 0.5], [-3.0, 1.5, 0.2]], dtype="float32")
        f1.update_state(y_true, logits)
        print(round(float(f1.result()), 2))  # 0.86
        ```
    """

    def __init__(self, average=None, threshold=0.5, from_logits=False, name="f1_score", dtype=None):
        super().__init__(average=average, threshold=threshold, name=name, dtype=dtype)
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = ops.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)

    def get_config(self):
        return {**super().get_config(), "from_logits": self.from_logits}
