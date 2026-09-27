"""Model wrappers: TransformedTargetModel and Potential interatomic potential."""

from __future__ import annotations

from typing import Optional, Union, Dict, Any
import keras
from keras import layers, ops
import numpy as np


class TransformedTargetModel(keras.Model):
    """Wraps a model and applies inverse transformation to predictions (e.g., mean/std denormalization)."""

    def __init__(
        self,
        model: keras.Model,
        mean: float = 0.0,
        std: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.mean = float(mean)
        self.std = float(std)

    def call(self, inputs, **kwargs):
        pred = self.model(inputs, **kwargs)
        return pred * self.std + self.mean


class Potential(keras.Model):
    """Interatomic potential wrapping an energy model and computing energies, forces, and stresses."""

    def __init__(
        self,
        model: keras.Model,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        element_refs: Optional[Dict[int, float]] = None,
        calc_forces: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.data_mean = float(data_mean)
        self.data_std = float(data_std)
        self.element_refs = element_refs or {}
        self.calc_forces = calc_forces

    def call(self, inputs, edge_index=None, node_type=None, **kwargs):
        e_pred = self.model(inputs, edge_index=edge_index, node_type=node_type, **kwargs)
        e_total = e_pred * self.data_std + self.data_mean
        return e_total
