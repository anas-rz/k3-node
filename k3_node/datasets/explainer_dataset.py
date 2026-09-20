from typing import Any, Callable, Dict, Optional, Union
import numpy as np
from keras import ops

from k3_node.data import Data, InMemoryDataset
from k3_node.datasets.graph_generator import GraphGenerator
from k3_node.datasets.motif_generator import MotifGenerator


class ExplainerDataset(InMemoryDataset):
    r"""Generates a synthetic dataset for evaluating explainability algorithms,
    as described in the "GNNExplainer: Generating Explanations for Graph Neural Networks" paper.

    Args:
        graph_generator (GraphGenerator or str): The graph generator to use.
        motif_generator (MotifGenerator or str): The motif generator to use.
        num_motifs (int): The number of motifs to attach to the graph.
        num_graphs (int, optional): The number of graphs to generate. (default: 1)
        graph_generator_kwargs (dict, optional): Keyword arguments for graph generator.
        motif_generator_kwargs (dict, optional): Keyword arguments for motif generator.
        transform (callable, optional): Transform function.
    """

    def __init__(
        self,
        graph_generator: Union[GraphGenerator, str],
        motif_generator: Union[MotifGenerator, str],
        num_motifs: int,
        num_graphs: int = 1,
        graph_generator_kwargs: Optional[Dict[str, Any]] = None,
        motif_generator_kwargs: Optional[Dict[str, Any]] = None,
        transform: Optional[Callable] = None,
    ):
        super().__init__(root=None, transform=transform)

        if num_motifs <= 0:
            raise ValueError(f"At least one motif needs to be attached (got {num_motifs})")

        self.graph_generator = GraphGenerator.resolve(
            graph_generator, **(graph_generator_kwargs or {})
        )
        self.motif_generator = MotifGenerator.resolve(
            motif_generator, **(motif_generator_kwargs or {})
        )
        self.num_motifs = num_motifs

        data_list = [self.get_graph() for _ in range(num_graphs)]
        self.data, self.slices = self.collate(data_list)

    def get_graph(self) -> Data:
        data = self.graph_generator()
        edge_index_np = ops.convert_to_numpy(data.edge_index)
        num_nodes = data.num_nodes
        num_edges = edge_index_np.shape[1]

        edge_indices = [edge_index_np]
        node_masks = [np.zeros(num_nodes, dtype=np.float32)]
        edge_masks = [np.zeros(num_edges, dtype=np.float32)]
        ys = [np.zeros(num_nodes, dtype=np.int64)]

        connecting_nodes = np.random.permutation(num_nodes)[: self.num_motifs]
        for i in connecting_nodes.tolist():
            motif = self.motif_generator()
            motif_ei = ops.convert_to_numpy(motif.edge_index)
            motif_num_nodes = motif.num_nodes
            motif_num_edges = motif_ei.shape[1]

            edge_indices.append(motif_ei + num_nodes)
            node_masks.append(np.ones(motif_num_nodes, dtype=np.float32))
            edge_masks.append(np.ones(motif_num_edges, dtype=np.float32))

            j = int(np.random.randint(0, motif_num_nodes)) + num_nodes
            edge_indices.append(np.array([[i, j], [j, i]], dtype=np.int64))
            edge_masks.append(np.zeros(2, dtype=np.float32))

            if hasattr(motif, "y") and motif.y is not None:
                motif_y = ops.convert_to_numpy(motif.y)
                if np.min(motif_y) == 0:
                    ys.append(motif_y + 1)
                else:
                    ys.append(motif_y)
            else:
                ys.append(np.ones(motif_num_nodes, dtype=np.int64))

            num_nodes += motif_num_nodes

        return Data(
            edge_index=ops.convert_to_tensor(np.concatenate(edge_indices, axis=1), dtype="int64"),
            y=ops.convert_to_tensor(np.concatenate(ys, axis=0), dtype="int64"),
            edge_mask=ops.convert_to_tensor(np.concatenate(edge_masks, axis=0), dtype="float32"),
            node_mask=ops.convert_to_tensor(np.concatenate(node_masks, axis=0), dtype="float32"),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({len(self)}, "
            f"graph_generator={self.graph_generator}, "
            f"motif_generator={self.motif_generator}, "
            f"num_motifs={self.num_motifs})"
        )

