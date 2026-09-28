from typing import Optional

import numpy as np
from keras import ops

try:
    import torch
    import torch.utils.data
    BaseDataLoader = torch.utils.data.DataLoader
except ImportError:
    torch = None
    BaseDataLoader = object

from k3_node.data import Data
from k3_node.loader.keras_dataset import loader_bases


def _np(x):
    return np.asarray(ops.convert_to_numpy(x))


class _SamplingSteps:
    """The sampler's "dataset": item ``i`` is a freshly sampled ``(node_idx, edge_idx)`` pair."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __len__(self):
        return self.sampler.num_steps

    def __getitem__(self, idx):
        return self.sampler._sample_subgraph()


class GraphSAINTSampler(*loader_bases(BaseDataLoader)):
    r"""The GraphSAINT sampler base class from the `"GraphSAINT: Graph Sampling Based Inductive
    Learning Method" <https://arxiv.org/abs/1907.04931>`_ paper. Every step samples a set of
    nodes and yields the subgraph they induce.

    With ``sample_coverage > 0``, normalization statistics are estimated beforehand by sampling
    about ``sample_coverage`` times every node: ``node_norm`` (to weight each node's loss) and
    ``edge_norm`` (to weight each edge's message) are added to every subgraph.

    Args:
        data (Data): The graph data object.
        batch_size (int): The approximate number of samples per batch (see the subclasses).
        num_steps (int, optional): The number of iterations per epoch. (default: ``1``)
        sample_coverage (int): How many samples per node to compute the normalization
            statistics with; ``0`` skips them. (default: ``0``)
        save_dir (str, optional): Unused; kept for API compatibility with PyG.
        log (bool, optional): Unused; kept for API compatibility with PyG.
        **kwargs (optional): Additional arguments of :class:`torch.utils.data.DataLoader`.
    """

    def __init__(self, data: Data, batch_size: int, num_steps: int = 1, sample_coverage: int = 0,
                 save_dir: Optional[str] = None, log: bool = True, **kwargs):
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)
        kwargs.pop('shuffle', None)
        assert data.edge_index is not None

        self.num_steps = num_steps
        self._batch_size = batch_size
        self.sample_coverage = sample_coverage
        self.save_dir = save_dir
        self.log = log
        self.data = data
        self.N = data.num_nodes
        edge_index = _np(data.edge_index).astype(np.int64)
        self.E = edge_index.shape[1]
        # CSR by source node: the edges leaving node i are perm[rowptr[i]:rowptr[i+1]]
        self._row, self._col = edge_index
        self._perm = np.argsort(self._row, kind="stable")
        self._rowptr = np.concatenate([[0], np.cumsum(np.bincount(self._row, minlength=self.N))])

        steps = _SamplingSteps(self)
        if torch is not None:
            super().__init__(steps, batch_size=1, collate_fn=self._collate, **kwargs)
        else:
            self.dataset = steps
            self.batch_size = 1
            self.collate_fn = self._collate

        if self.sample_coverage > 0:
            self.node_norm, self.edge_norm = self._compute_norm()

    def _sample_nodes(self, batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def _sample_subgraph(self):
        node_idx = np.unique(self._sample_nodes(self._batch_size))
        in_sample = np.zeros(self.N, dtype=bool)
        in_sample[node_idx] = True
        edge_idx = np.nonzero(in_sample[self._row] & in_sample[self._col])[0]
        return node_idx, edge_idx

    def _collate(self, data_list):
        node_idx, edge_idx = data_list[0]
        new_id = np.full(self.N, -1, dtype=np.int64)
        new_id[node_idx] = np.arange(len(node_idx))

        data = Data()
        data.num_nodes = len(node_idx)
        data.edge_index = ops.convert_to_tensor(
            np.stack([new_id[self._row[edge_idx]], new_id[self._col[edge_idx]]]), dtype="int64")
        for key, item in self.data.items():
            if key in ('edge_index', 'num_nodes'):
                continue
            shape = getattr(item, 'shape', None)
            if shape is not None and len(shape) > 0 and shape[0] == self.N:
                data[key] = ops.take(item, node_idx, axis=0)
            elif shape is not None and len(shape) > 0 and shape[0] == self.E:
                data[key] = ops.take(item, edge_idx, axis=0)
            else:
                data[key] = item
        if self.sample_coverage > 0:
            data.node_norm = ops.convert_to_tensor(self.node_norm[node_idx])
            data.edge_norm = ops.convert_to_tensor(self.edge_norm[edge_idx])
        return data

    def _compute_norm(self):
        node_count = np.zeros(self.N, dtype=np.float32)
        edge_count = np.zeros(self.E, dtype=np.float32)
        num_samples = total_sampled_nodes = 0
        while total_sampled_nodes < self.N * self.sample_coverage:
            for _ in range(self.num_steps):
                node_idx, edge_idx = self._sample_subgraph()
                node_count[node_idx] += 1
                edge_count[edge_idx] += 1
                total_sampled_nodes += len(node_idx)
            num_samples += self.num_steps

        with np.errstate(divide='ignore', invalid='ignore'):
            edge_norm = np.clip(node_count[self._row] / edge_count, 0, 1e4)
        edge_norm[np.isnan(edge_norm)] = 0.1
        node_count[node_count == 0] = 0.1
        node_norm = num_samples / node_count / self.N
        return node_norm.astype(np.float32), edge_norm.astype(np.float32)

    def _random_walk(self, start: np.ndarray, walk_length: int) -> np.ndarray:
        walks, cur = [start], start
        for _ in range(walk_length):
            deg = self._rowptr[cur + 1] - self._rowptr[cur]
            offset = np.floor(np.random.rand(len(cur)) * np.maximum(deg, 1)).astype(np.int64)
            nxt = self._col[self._perm[np.minimum(self._rowptr[cur] + offset, self.E - 1)]]
            cur = np.where(deg > 0, nxt, cur)  # nodes without neighbors stay put
            walks.append(cur)
        return np.stack(walks, axis=1)


class GraphSAINTNodeSampler(GraphSAINTSampler):
    r"""The GraphSAINT node sampler: samples ``batch_size`` nodes, each with probability
    proportional to its out-degree."""

    def _sample_nodes(self, batch_size: int) -> np.ndarray:
        return self._row[np.random.randint(0, self.E, size=batch_size)]


class GraphSAINTEdgeSampler(GraphSAINTSampler):
    r"""The GraphSAINT edge sampler: samples ``batch_size`` edges, each with probability
    proportional to :math:`1 / \deg(u) + 1 / \deg(v)`, and keeps their endpoints."""

    def _sample_nodes(self, batch_size: int) -> np.ndarray:
        out_deg = np.maximum(np.bincount(self._row, minlength=self.N), 1)
        in_deg = np.maximum(np.bincount(self._col, minlength=self.N), 1)
        prob = 1.0 / in_deg[self._row] + 1.0 / out_deg[self._col]
        # Weighted sampling without replacement (exponential keys, as in PyG)
        keys = np.log(np.random.rand(self.E)) / (prob + 1e-10)
        edge_sample = np.argsort(-keys)[:batch_size]
        return np.concatenate([self._col[edge_sample], self._row[edge_sample]])


class GraphSAINTRandomWalkSampler(GraphSAINTSampler):
    r"""The GraphSAINT random walk sampler: starts ``batch_size`` random walks of length
    ``walk_length`` and keeps the visited nodes.

    Args:
        walk_length (int): Length of each random walk.
    """

    def __init__(self, data: Data, batch_size: int, walk_length: int, num_steps: int = 1,
                 sample_coverage: int = 0, save_dir: Optional[str] = None, log: bool = True, **kwargs):
        self.walk_length = walk_length
        super().__init__(data, batch_size, num_steps, sample_coverage, save_dir, log, **kwargs)

    def _sample_nodes(self, batch_size: int) -> np.ndarray:
        start = np.random.randint(0, self.N, size=batch_size)
        return self._random_walk(start, self.walk_length).reshape(-1)
