from typing import Any, List, Optional
import numpy as np
from keras import ops

from k3_node.data.data import Data
from k3_node.data.storage import is_tensor_like


class HypergraphData(Data):
    """A data object describing a hypergraph."""

    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        pos=None,
        **kwargs,
    ):
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            **kwargs,
        )

    @property
    def num_edges(self) -> int:
        if self.edge_index is None:
            return 0
        ei_np = ops.convert_to_numpy(self.edge_index)
        if ei_np.size == 0 or ei_np.shape[1] == 0:
            return 0
        return int(np.max(ei_np[1])) + 1

    @property
    def num_nodes(self) -> Optional[int]:
        num = super().num_nodes
        if self.edge_index is not None and num == self.num_edges:
            ei_np = ops.convert_to_numpy(self.edge_index)
            if ei_np.size > 0 and ei_np.shape[1] > 0:
                return int(np.max(ei_np[0])) + 1
        return num

    @num_nodes.setter
    def num_nodes(self, num_nodes: Optional[int]) -> None:
        self._store.num_nodes = num_nodes

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if key == "edge_index":
            return np.array([[self.num_nodes], [self.num_edges]])
        return super().__inc__(key, value, *args, **kwargs)


HyperGraphData = HypergraphData

