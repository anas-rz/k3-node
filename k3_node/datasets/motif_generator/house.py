import numpy as np
from keras import ops

from k3_node.data import Data
from k3_node.datasets.motif_generator.custom import CustomMotif


class HouseMotif(CustomMotif):
    r"""Generates the house-structured motif from the "GNNExplainer" paper,
    containing 5 nodes and 6 undirected edges.
    """

    def __init__(self):
        edge_index = ops.convert_to_tensor(
            np.array(
                [
                    [0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4],
                    [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1],
                ],
                dtype=np.int64,
            ),
            dtype="int64",
        )
        y = ops.convert_to_tensor(np.array([0, 0, 1, 1, 2], dtype=np.int64), dtype="int64")
        structure = Data(num_nodes=5, edge_index=edge_index, y=y)
        super().__init__(structure)

