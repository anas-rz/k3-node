from abc import ABC, abstractmethod
from typing import Any
from k3_node.data import Data


class MotifGenerator(ABC):
    r"""An abstract base class for generating motifs."""

    @abstractmethod
    def __call__(self) -> Data:
        raise NotImplementedError

    @staticmethod
    def resolve(query: Any, *args: Any, **kwargs: Any) -> "MotifGenerator":
        if isinstance(query, MotifGenerator):
            return query
        if isinstance(query, str):
            query = query.lower()
            if query in ["house", "housemotif"]:
                from k3_node.datasets.motif_generator.house import HouseMotif
                return HouseMotif(*args, **kwargs)
            elif query in ["cycle", "cyclemotif"]:
                from k3_node.datasets.motif_generator.cycle import CycleMotif
                return CycleMotif(*args, **kwargs)
        raise ValueError(f"Could not resolve motif generator: {query}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

