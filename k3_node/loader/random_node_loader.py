import math
from typing import Union

import numpy as np

try:
    import torch
    import torch.utils.data
    from torch import Tensor
    BaseDataLoader = torch.utils.data.DataLoader
except ImportError:
    torch = None
    Tensor = type(None)
    BaseDataLoader = object

from k3_node.data import Data, HeteroData


class RandomNodeLoader(BaseDataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    Args:
        data (Data or HeteroData): The graph data object.
        num_parts (int): The number of partitions.
        **kwargs (optional): Additional arguments of :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_parts: int,
        **kwargs,
    ):
        self.data = data
        self.num_parts = num_parts

        if isinstance(data, HeteroData):
            node_dict = {}
            total = 0
            for node_type in data.node_types:
                count = data[node_type].num_nodes
                node_dict[node_type] = (total, total + count)
                total += count
            self.node_dict = node_dict
            self.num_nodes = total
        else:
            self.edge_index = data.edge_index
            self.num_nodes = data.num_nodes

        kwargs.pop('dataset', None)
        kwargs.pop('batch_size', None)
        kwargs.pop('collate_fn', None)

        batch_size = math.ceil(self.num_nodes / num_parts) if num_parts > 0 else self.num_nodes

        if torch is not None:
            super().__init__(
                range(self.num_nodes),
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                **kwargs,
            )
        else:
            self.dataset = range(self.num_nodes)
            self.batch_size = batch_size
            self.collate_fn = self.collate_fn

    def collate_fn(self, index):
        if torch is not None and not isinstance(index, Tensor):
            index = torch.tensor(index, dtype=torch.long)
        elif torch is None:
            index = np.asarray(index, dtype=np.int64)

        if isinstance(self.data, Data):
            return self.data.subgraph(index)

        elif isinstance(self.data, HeteroData):
            node_dict = {}
            for key, (start, end) in self.node_dict.items():
                if torch is not None and isinstance(index, Tensor):
                    mask = (index >= start) & (index < end)
                    node_dict[key] = index[mask] - start
                else:
                    idx_np = np.asarray(index)
                    mask = (idx_np >= start) & (idx_np < end)
                    node_dict[key] = idx_np[mask] - start
            return self.data.subgraph(node_dict)

