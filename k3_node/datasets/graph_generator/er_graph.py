from k3_node.data import Data
from k3_node.datasets.graph_generator.base import GraphGenerator
from k3_node.utils.random import erdos_renyi_graph


class ERGraph(GraphGenerator):
    r"""Generates random Erdos-Renyi (ER) graphs."""

    def __init__(self, num_nodes: int, edge_prob: float, directed: bool = False):
        super().__init__()
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.directed = directed

    def __call__(self) -> Data:
        edge_index = erdos_renyi_graph(self.num_nodes, self.edge_prob, directed=self.directed)
        return Data(num_nodes=self.num_nodes, edge_index=edge_index)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_nodes={self.num_nodes}, edge_prob={self.edge_prob})"

