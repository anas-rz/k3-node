"""Model wrappers: TransformedTargetModel and Potential interatomic potential."""

from __future__ import annotations

from typing import Optional, Union, Dict, Any
import keras
from keras import layers, ops
import numpy as np


class TransformedTargetModel(keras.Model):
    """Wraps a model and applies inverse transformation to predictions (e.g., mean/std denormalization).

    Example:
        ```python
        import numpy as np
        from k3_node.models import MEGNet, TransformedTargetModel

        # A 4-atom structure: positions, bonds (listed in both directions) and atomic numbers
        structure = {
            "pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.2, 0.8], [1.5, 1.5, 1.0]], dtype="float32"),
            "edge_index": np.array([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]),
            "line_edge_index": np.array([[0, 1, 2, 3], [1, 2, 3, 0]]),  # bond pairs forming angles
            "node_type": np.array([6, 8, 1, 6]),  # atomic numbers
            "batch": np.zeros(4, dtype="int32"),  # all atoms belong to structure 0
            "state_attr": np.zeros((1, 2), dtype="float32"),  # global state features
        }

        base = MEGNet(dim_node_embedding=8, dim_edge_embedding=16, nblocks=1)
        model = TransformedTargetModel(model=base, mean=5.0, std=2.0)  # outputs base * std + mean
        print(tuple(model(structure).shape))  # (1,)
        ```
    """

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

    def call(self, inputs, training=None, **kwargs):
        pred = self.model(inputs, training=training, **kwargs)
        return pred * self.std + self.mean


class Potential(keras.Model):
    """Interatomic potential wrapping an energy model and computing energies, forces, and stresses.

    Example:
        ```python
        import numpy as np
        from k3_node.models import MEGNet, Potential

        # A 4-atom structure: positions, bonds (listed in both directions) and atomic numbers
        structure = {
            "pos": np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.5, 1.2, 0.8], [1.5, 1.5, 1.0]], dtype="float32"),
            "edge_index": np.array([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]]),
            "line_edge_index": np.array([[0, 1, 2, 3], [1, 2, 3, 0]]),  # bond pairs forming angles
            "node_type": np.array([6, 8, 1, 6]),  # atomic numbers
            "batch": np.zeros(4, dtype="int32"),  # all atoms belong to structure 0
            "state_attr": np.zeros((1, 2), dtype="float32"),  # global state features
        }

        base = MEGNet(dim_node_embedding=8, dim_edge_embedding=16, nblocks=1)
        potential = Potential(model=base, data_mean=-1.5, data_std=0.8)  # interatomic potential wrapper
        print(tuple(potential(structure).shape))  # (1,)
        ```
    """

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

    def call(self, inputs, edge_index=None, node_type=None, training=None, **kwargs):
        e_pred = self.model(inputs, edge_index=edge_index, node_type=node_type, training=training, **kwargs)
        e_total = e_pred * self.data_std + self.data_mean
        return e_total
