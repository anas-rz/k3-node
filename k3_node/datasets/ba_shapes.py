from typing import Callable, Optional, Tuple
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.utils.random import barabasi_albert_graph


def house() -> Tuple[np.ndarray, np.ndarray]:
    edge_index = np.array(
        [[0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4], [1, 3, 4, 4, 2, 0, 1, 3, 2, 0, 0, 1]],
        dtype=np.int64,
    )
    label = np.array([1, 1, 2, 2, 3], dtype=np.int64)
    return edge_index, label


class BAShapes(InMemoryDataset):
    r"""The BA-Shapes dataset from the "GNNExplainer: Generating Explanations
    for Graph Neural Networks" paper, containing a Barabasi-Albert (BA) graph
    with 300 nodes and a set of 80 "house"-structured graphs connected to it.

    Args:
        connection_distribution (str, optional): Specifies how the houses and
            the BA graph get connected ("random", "uniform"). (default: "random")
        transform (callable, optional): Transform function.
    """

    def __init__(
        self,
        connection_distribution: str = "random",
        transform: Optional[Callable] = None,
    ):
        super().__init__(None, transform)
        assert connection_distribution in ["random", "uniform"]

        num_nodes = 300
        edge_index = ops.convert_to_numpy(barabasi_albert_graph(num_nodes, num_edges=5))
        edge_label = np.zeros(edge_index.shape[1], dtype=np.int64)
        node_label = np.zeros(num_nodes, dtype=np.int64)

        num_houses = 80
        if connection_distribution == "random":
            connecting_nodes = np.random.permutation(num_nodes)[:num_houses]
        else:
            step = num_nodes // num_houses
            connecting_nodes = np.arange(0, num_nodes, step)

        edge_indices = [edge_index]
        edge_labels = [edge_label]
        node_labels = [node_label]

        for i in range(num_houses):
            house_edge_index, house_label = house()
            edge_indices.append(house_edge_index + num_nodes)
            edge_indices.append(
                np.array([[int(connecting_nodes[i]), num_nodes], [num_nodes, int(connecting_nodes[i])]], dtype=np.int64)
            )
            edge_labels.append(np.ones(house_edge_index.shape[1], dtype=np.int64))
            edge_labels.append(np.zeros(2, dtype=np.int64))
            node_labels.append(house_label)
            num_nodes += 5

        edge_index = np.concatenate(edge_indices, axis=1)
        edge_label = np.concatenate(edge_labels, axis=0)
        node_label = np.concatenate(node_labels, axis=0)

        x = np.ones((num_nodes, 10), dtype=np.float32)
        expl_mask = np.zeros(num_nodes, dtype=bool)
        expl_mask[np.arange(400, num_nodes, 5)] = True

        data = Data(
            x=ops.convert_to_tensor(x, dtype="float32"),
            edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
            y=ops.convert_to_tensor(node_label, dtype="int64"),
            expl_mask=ops.convert_to_tensor(expl_mask, dtype="bool"),
            edge_label=ops.convert_to_tensor(edge_label, dtype="int64"),
        )

        self.data, self.slices = self.collate([data])

