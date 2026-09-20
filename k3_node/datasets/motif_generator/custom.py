from typing import Any, Optional
from k3_node.data import Data
from k3_node.datasets.motif_generator.base import MotifGenerator


class CustomMotif(MotifGenerator):
    r"""Generates a motif based on a custom structure coming from a Data object."""

    def __init__(self, structure: Any):
        super().__init__()
        if not isinstance(structure, Data):
            raise ValueError(f"Expected structure of type Data, got {type(structure)}")
        self.structure = structure

    def __call__(self) -> Data:
        return self.structure

