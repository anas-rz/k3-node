from abc import ABC, abstractmethod
from typing import Any
from k3_node.data import Data


class GraphGenerator(ABC):
    r"""An abstract base class for generating synthetic graphs."""

    @abstractmethod
    def __call__(self) -> Data:
        raise NotImplementedError

    @staticmethod
    def resolve(query: Any, *args: Any, **kwargs: Any) -> "GraphGenerator":
        if isinstance(query, GraphGenerator):
            return query
        if isinstance(query, str):
            query = query.lower()
            if query in ["ba", "bagraph", "barabasi_albert"]:
                from k3_node.datasets.graph_generator.ba_graph import BAGraph
                return BAGraph(*args, **kwargs)
            elif query in ["er", "ergraph", "erdos_renyi"]:
                from k3_node.datasets.graph_generator.er_graph import ERGraph
                return ERGraph(*args, **kwargs)
        raise ValueError(f"Could not resolve graph generator: {query}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

