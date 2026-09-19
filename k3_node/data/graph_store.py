from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class EdgeLayout(Enum):
    COO = "coo"
    CSC = "csc"
    CSR = "csr"


@dataclass
class EdgeAttr:
    """Defines the attributes of a GraphStore edge."""

    edge_type: Any
    layout: EdgeLayout
    is_sorted: bool = False
    size: Optional[Tuple[int, int]] = None

    def __init__(
        self,
        edge_type: Any,
        layout: Union[EdgeLayout, str] = EdgeLayout.COO,
        is_sorted: bool = False,
        size: Optional[Tuple[int, int]] = None,
    ):
        if isinstance(layout, str):
            layout = EdgeLayout(layout.lower())
        self.edge_type = edge_type
        self.layout = layout
        self.is_sorted = is_sorted
        self.size = size


class GraphStore(ABC):
    """Abstract base class for graph edge stores."""

    @abstractmethod
    def _put_edge_index(self, edge_index: Any, edge_attr: EdgeAttr) -> bool:
        pass

    @abstractmethod
    def _get_edge_index(self, edge_attr: EdgeAttr) -> Optional[Any]:
        pass

    @abstractmethod
    def _remove_edge_index(self, edge_attr: EdgeAttr) -> bool:
        pass

    def put_edge_index(
        self,
        edge_index: Any,
        edge_type: Any,
        layout: Union[EdgeLayout, str] = EdgeLayout.COO,
        is_sorted: bool = False,
        size: Optional[Tuple[int, int]] = None,
    ) -> bool:
        attr = EdgeAttr(edge_type=edge_type, layout=layout, is_sorted=is_sorted, size=size)
        return self._put_edge_index(edge_index, attr)

    def get_edge_index(
        self,
        edge_type: Any,
        layout: Union[EdgeLayout, str] = EdgeLayout.COO,
        is_sorted: bool = False,
    ) -> Optional[Any]:
        attr = EdgeAttr(edge_type=edge_type, layout=layout, is_sorted=is_sorted)
        return self._get_edge_index(attr)

    def remove_edge_index(
        self,
        edge_type: Any,
        layout: Union[EdgeLayout, str] = EdgeLayout.COO,
    ) -> bool:
        attr = EdgeAttr(edge_type=edge_type, layout=layout)
        return self._remove_edge_index(attr)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2:
            edge_type, layout = key
            return self.get_edge_index(edge_type=edge_type, layout=layout)
        return self.get_edge_index(edge_type=key)

    def __setitem__(self, key: Any, value: Any):
        if isinstance(key, tuple) and len(key) == 2:
            edge_type, layout = key
            self.put_edge_index(value, edge_type=edge_type, layout=layout)
        else:
            self.put_edge_index(value, edge_type=key)

