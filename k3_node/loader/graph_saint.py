from typing import Optional

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

from k3_node.data import Data
from k3_node.loader.sampler_utils import FastGraph, random_walk


class GraphSAINTSampler(BaseDataLoader):
    r"""The GraphSAINT sampler base class from the "GraphSAINT" paper.

    Args:
        data (Data): The graph data object.
        batch_size (int): Approximate number of nodes per batch.
        num_steps (int, optional): Number of iterations per epoch. (default: :obj:`1`)
        sample_coverage (int): Coverage for normalization statistics. (default: :obj:`0`)
        save_dir (str, optional): Directory to save normalization stats. (default: :obj:`None`)
        log (bool, optional): Logging flag. (default: :obj:`True`)
    """
    def __init__(
        self,
        data: Data,
        batch_size: int,
        num_steps: int = 1,
        sample_coverage: int = 0,
        save_dir: Optional[str] = None,
        log: bool = True,
        **kwargs,
    ):
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        assert data.edge_index is not None
        self.num_steps = num_steps
        self._batch_size = batch_size
        self.sample_coverage = sample_coverage
        self.save_dir = save_dir
        self.log = log

        self.N = data.num_nodes
        self.E = data.num_edges
        self.data = data
        self.graph = FastGraph(data.edge_index, num_nodes=self.N)

        if torch is not None:
            super().__init__(self, batch_size=1, collate_fn=self._collate, **kwargs)
        else:
            self.dataset = self
            self.batch_size = 1
            self.collate_fn = self._collate

        if self.sample_coverage > 0:
            self.node_norm, self.edge_norm = self._compute_norm()

    def __len__(self) -> int:
        return self.num_steps

    def _sample_nodes(self, batch_size: int):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        node_idx = self._sample_nodes(self._batch_size)
        if torch is not None and isinstance(node_idx, Tensor):
            node_idx = node_idx.unique()
        else:
            node_idx = np.unique(np.asarray(node_idx))
        return node_idx

    def _collate(self, data_list):
        node_idx = data_list[0]
        subgraph = self.data.subgraph(node_idx)

        if self.sample_coverage > 0:
            subgraph.node_norm = self.node_norm[node_idx]
            if hasattr(self, 'edge_norm') and subgraph.edge_index is not None:
                subgraph.edge_norm = self.edge_norm[:subgraph.edge_index.shape[1]]

        return subgraph

    def _compute_norm(self):
        is_torch = torch is not None and isinstance(self.data.edge_index, Tensor)
        if is_torch:
            node_count = torch.zeros(self.N, dtype=torch.float, device=self.data.edge_index.device)
            edge_count = torch.zeros(self.E, dtype=torch.float, device=self.data.edge_index.device)
            node_count[node_count == 0] = 1.0
            edge_count[edge_count == 0] = 1.0
            node_norm = self.num_steps / (node_count * self.N)
            edge_norm = 1.0 / edge_count
            return node_norm, edge_norm
        else:
            node_norm = np.ones(self.N, dtype=np.float32) / self.N
            edge_norm = np.ones(self.E, dtype=np.float32)
            return node_norm, edge_norm


class GraphSAINTNodeSampler(GraphSAINTSampler):
    r"""The GraphSAINT node sampler class."""
    def _sample_nodes(self, batch_size: int):
        is_torch = torch is not None and isinstance(self.data.edge_index, Tensor)
        edge_sample_size = min(batch_size, self.E)
        if is_torch:
            edge_sample = torch.randint(0, self.E, (edge_sample_size,), dtype=torch.long, device=self.data.edge_index.device)
            return self.data.edge_index[0, edge_sample]
        else:
            edge_sample = np.random.randint(0, self.E, size=edge_sample_size)
            return np.asarray(self.data.edge_index)[0, edge_sample]


class GraphSAINTEdgeSampler(GraphSAINTSampler):
    r"""The GraphSAINT edge sampler class."""
    def _sample_nodes(self, batch_size: int):
        is_torch = torch is not None and isinstance(self.data.edge_index, Tensor)
        edge_sample_size = min(batch_size, self.E)
        if is_torch:
            edge_sample = torch.randint(0, self.E, (edge_sample_size,), dtype=torch.long, device=self.data.edge_index.device)
            endpoints = torch.cat([self.data.edge_index[0, edge_sample], self.data.edge_index[1, edge_sample]], dim=0)
            return endpoints
        else:
            edge_sample = np.random.randint(0, self.E, size=edge_sample_size)
            np_edges = np.asarray(self.data.edge_index)
            return np.concatenate([np_edges[0, edge_sample], np_edges[1, edge_sample]], axis=0)


class GraphSAINTRandomWalkSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler class.

    Args:
        walk_length (int): Length of each random walk.
    """
    def __init__(
        self,
        data: Data,
        batch_size: int,
        walk_length: int,
        num_steps: int = 1,
        sample_coverage: int = 0,
        save_dir: Optional[str] = None,
        log: bool = True,
        **kwargs,
    ):
        self.walk_length = walk_length
        super().__init__(data, batch_size, num_steps, sample_coverage, save_dir, log, **kwargs)

    def _sample_nodes(self, batch_size: int):
        is_torch = torch is not None and isinstance(self.data.edge_index, Tensor)
        num_walks = max(1, batch_size // self.walk_length)
        if is_torch:
            start_nodes = torch.randint(0, self.N, (num_walks,), dtype=torch.long, device=self.data.edge_index.device)
            walks = random_walk(self.data.edge_index, start_nodes, self.walk_length, num_nodes=self.N)
            return walks.view(-1)
        else:
            start_nodes = np.random.randint(0, self.N, size=num_walks)
            walks = random_walk(self.data.edge_index, start_nodes, self.walk_length, num_nodes=self.N)
            return walks.reshape(-1)
