import copy
from typing import Any, List, Optional, Union

import numpy as np

try:
    import torch
    import torch.utils.data
    from torch import Tensor
    BaseDataset = torch.utils.data.Dataset
    BaseDataLoader = torch.utils.data.DataLoader
except ImportError:
    torch = None
    Tensor = type(None)
    BaseDataset = object
    BaseDataLoader = object

from k3_node.data import Data
from k3_node.loader.sampler_utils import partition_graph


class ClusterData(BaseDataset):
    r"""Clusters/partitions a graph data object into multiple subgraphs, as
    motivated by the "Cluster-GCN" paper.

    Args:
        data (Data): The graph data object.
        num_parts (int): The number of partitions.
        recursive (bool, optional): Multilevel recursive bisection if True. (default: :obj:`False`)
        save_dir (str, optional): Directory to save partitioned data. (default: :obj:`None`)
        log (bool, optional): If set to :obj:`False`, will not log. (default: :obj:`True`)
        keep_inter_cluster_edges (bool, optional): Keep inter-cluster connections. (default: :obj:`False`)
    """
    def __init__(
        self,
        data: Data,
        num_parts: int,
        recursive: bool = False,
        save_dir: Optional[str] = None,
        filename: Optional[str] = None,
        log: bool = True,
        keep_inter_cluster_edges: bool = False,
        sparse_format: str = 'csr',
    ):
        assert data.edge_index is not None

        self.num_parts = num_parts
        self.recursive = recursive
        self.keep_inter_cluster_edges = keep_inter_cluster_edges
        self.sparse_format = sparse_format
        self.data = data

        self.cluster = partition_graph(data.edge_index, data.num_nodes, num_parts)

        # Precompute part indices
        self.part_nodes = []
        is_torch = torch is not None and isinstance(self.cluster, Tensor)
        for i in range(num_parts):
            if is_torch:
                nodes = (self.cluster == i).nonzero(as_tuple=False).view(-1)
            else:
                nodes = np.nonzero(self.cluster == i)[0]
            self.part_nodes.append(nodes)

    def __len__(self) -> int:
        return self.num_parts

    def __getitem__(self, idx: int) -> Data:
        nodes = self.part_nodes[idx]
        return self.data.subgraph(nodes)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_parts})'


class ClusterLoader(BaseDataLoader):
    r"""The data loader scheme from Cluster-GCN which merges partitioned
    subgraphs to form a mini-batch.

    Args:
        cluster_data (ClusterData): The already partitioned data object.
        **kwargs (optional): Additional arguments of :class:`torch.utils.data.DataLoader`.
    """
    def __init__(self, cluster_data: ClusterData, **kwargs):
        self.cluster_data = cluster_data
        kwargs.pop('collate_fn', None)
        iterator = range(len(cluster_data))

        if torch is not None:
            super().__init__(iterator, collate_fn=self._collate, **kwargs)
        else:
            self.dataset = iterator
            self.collate_fn = self._collate

    def _collate(self, batch: List[int]) -> Data:
        all_nodes = []
        is_torch = torch is not None and isinstance(self.cluster_data.cluster, Tensor)

        for part_id in batch:
            all_nodes.append(self.cluster_data.part_nodes[part_id])

        if is_torch:
            nodes = torch.cat(all_nodes, dim=0)
        else:
            nodes = np.concatenate([np.asarray(x) for x in all_nodes], axis=0)

        return self.cluster_data.data.subgraph(nodes)

