import numpy as np
from keras import ops

from k3_node.data import Data
from k3_node.datasets.motif_generator.custom import CustomMotif


class CycleMotif(CustomMotif):
    r"""Generates the cycle motif from the "GNNExplainer" paper."""

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

        row = np.repeat(np.arange(num_nodes), 2)
        col1 = np.mod(np.arange(-1, num_nodes - 1), num_nodes)
        col2 = np.mod(np.arange(1, num_nodes + 1), num_nodes)
        col = np.sort(np.stack([col1, col2], axis=1), axis=-1).flatten()

        edge_index = ops.convert_to_tensor(np.stack([row, col], axis=0), dtype="int64")
        structure = Data(num_nodes=num_nodes, edge_index=edge_index)
        super().__init__(structure)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.num_nodes})"

