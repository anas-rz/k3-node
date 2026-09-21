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

        loaded = False
        if save_dir is not None:
            import os.path as osp
            recursive_str = '_recursive' if recursive else ''
            root_dir = osp.join(save_dir, f'part_{num_parts}{recursive_str}')
            path = osp.join(root_dir, filename or 'metis.pt')
            if osp.exists(path):
                try:
                    import torch
                    part = torch.load(path, map_location="cpu", weights_only=False)
                    if hasattr(part, "partptr") and hasattr(part, "node_perm"):
                        partptr = np.asarray(part.partptr)
                        node_perm = np.asarray(part.node_perm)
                        self.part_nodes = [node_perm[partptr[i]:partptr[i+1]] for i in range(num_parts)]
                        self.cluster = np.zeros(data.num_nodes, dtype=np.int64)
                        for i in range(num_parts):
                            self.cluster[self.part_nodes[i]] = i
                        loaded = True
                except Exception:
                    pass

        if not loaded:
            self.cluster = partition_graph(data.edge_index, data.num_nodes, num_parts)
            from k3_node.loader.utils import to_numpy
            cluster_np = to_numpy(self.cluster)
            sort_idx = np.argsort(cluster_np)
            sorted_cluster = cluster_np[sort_idx]
            split_idx = np.searchsorted(sorted_cluster, np.arange(num_parts + 1))
            self.part_nodes = [sort_idx[split_idx[i]:split_idx[i+1]] for i in range(num_parts)]

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

        if is_torch and all_nodes and isinstance(all_nodes[0], Tensor):
            nodes = torch.cat(all_nodes, dim=0)
        else:
            nodes = np.concatenate([np.asarray(x) for x in all_nodes], axis=0)

        return self.cluster_data.data.subgraph(nodes)

