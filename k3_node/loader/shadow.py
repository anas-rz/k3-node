import copy
from typing import Any, List, Optional

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

from k3_node.data import Batch, Data
from k3_node.loader.sampler_utils import sample_neighbors_homo


class ShaDowKHopSampler(BaseDataLoader):
    r"""The ShaDow k-hop sampler from the "Decoupling the Depth and Scope of
    Graph Neural Networks" paper.

    Args:
        data (Data): The graph data object.
        depth (int): The depth/number of hops of the localized subgraph.
        num_neighbors (int): The number of neighbors to sample for each node in each hop.
        node_idx (LongTensor or BoolTensor, optional): Seed nodes. (default: :obj:`None`)
        replace (bool, optional): Sample neighbors with replacement. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        data: Data,
        depth: int,
        num_neighbors: int,
        node_idx: Optional[Any] = None,
        replace: bool = False,
        **kwargs,
    ):
        self.data = copy.copy(data)
        self.depth = depth
        self.num_neighbors = num_neighbors
        self.replace = replace

        if node_idx is None:
            if torch is not None and isinstance(data.edge_index, Tensor):
                node_idx = torch.arange(data.num_nodes, device=data.edge_index.device)
            else:
                node_idx = np.arange(data.num_nodes)
        elif torch is not None and isinstance(node_idx, Tensor) and node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        elif isinstance(node_idx, np.ndarray) and node_idx.dtype == bool:
            node_idx = np.nonzero(node_idx)[0]

        self.node_idx = node_idx
        idx_list = node_idx.tolist() if hasattr(node_idx, 'tolist') else list(node_idx)

        if torch is not None:
            super().__init__(idx_list, collate_fn=self.__collate__, **kwargs)
        else:
            self.dataset = idx_list
            self.collate_fn = self.__collate__

    def __collate__(self, n_id: List[int]) -> Batch:
        subgraphs = []
        is_torch = torch is not None and isinstance(self.data.edge_index, Tensor)
        num_neighbors_list = [self.num_neighbors] * self.depth

        for root in n_id:
            root_seeds = [root]
            if is_torch:
                root_seeds = torch.tensor(root_seeds, dtype=torch.long, device=self.data.edge_index.device)

            nodes, row, col, edges, _, _ = sample_neighbors_homo(
                edge_index=self.data.edge_index,
                seed_nodes=root_seeds,
                num_neighbors=num_neighbors_list,
                num_nodes=self.data.num_nodes,
                replace=self.replace,
                subgraph_type='directional',
            )

            sub = self.data.subgraph(nodes)
            sub.root_n_id = 0  # Root node is always the first node (seed node)
            subgraphs.append(sub)

        batch = Batch.from_data_list(subgraphs)
        if is_torch:
            batch.root_n_id = torch.zeros(len(n_id), dtype=torch.long, device=self.data.edge_index.device)
        else:
            batch.root_n_id = np.zeros(len(n_id), dtype=np.int64)

        return batch
