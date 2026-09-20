from k3_node.data import Data
from k3_node.datasets.graph_generator.base import GraphGenerator
from k3_node.utils.random import barabasi_albert_graph


class BAGraph(GraphGenerator):
    r"""Generates random Barabasi-Albert (BA) graphs."""

    def __init__(self, num_nodes: int, num_edges: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

    def __call__(self) -> Data:
        edge_index = barabasi_albert_graph(self.num_nodes, self.num_edges)
        return Data(num_nodes=self.num_nodes, edge_index=edge_index)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_nodes={self.num_nodes}, num_edges={self.num_edges})"

