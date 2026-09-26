"""Base task abstraction for high-level K3-Node estimators."""

from typing import Any, Dict, List, Optional, Tuple, Union
import keras
from keras import ops

from k3_node.data import BaseData


class BaseTask:
    r"""Abstract base task estimator providing common training, evaluation,
    and serialization workflows.
    """

    def __init__(self, model: Optional[keras.Model] = None):
        self.model = model
        self._is_compiled = False

    def compile(
        self,
        optimizer: Optional[Union[str, keras.optimizers.Optimizer]] = None,
        loss: Optional[Any] = None,
        metrics: Optional[List[Any]] = None,
        **kwargs,
    ):
        r"""Configures the task model for training."""
        if self.model is None:
            raise RuntimeError("Model has not been initialized. Call fit() or construct with a model first.")

        opt = optimizer or keras.optimizers.Adam(learning_rate=0.01)
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics, **kwargs)
        self._is_compiled = True
        return self

    def summary(self):
        r"""Prints a string summary of the underlying neural network."""
        if self.model is not None:
            return self.model.summary()
        print("Model has not been initialized yet.")

    def save(self, filepath: str):
        r"""Saves the underlying model weights."""
        if self.model is not None:
            self.model.save(filepath)
        else:
            raise RuntimeError("Cannot save an uninitialized model.")

    @classmethod
    def load(cls, filepath: str, **kwargs):
        r"""Loads a saved task model from disk."""
        model = keras.models.load_model(filepath, **kwargs)
        instance = cls(model=model)
        instance._is_compiled = True
        return instance

    def _extract_inputs(self, data: Any):
        r"""Extracts input tensors from a Data object, tuple, or dictionary."""
        if hasattr(data, "inputs"):
            return data.inputs
        elif isinstance(data, (tuple, list)):
            return data
        elif hasattr(data, "x") and hasattr(data, "edge_index"):
            if hasattr(data, "edge_attr") and data.edge_attr is not None:
                return (data.x, data.edge_index, data.edge_attr)
            return (data.x, data.edge_index)
        return data
