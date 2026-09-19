from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor
    BaseDataLoader = torch.utils.data.DataLoader
except ImportError:
    torch = None
    Tensor = type(None)
    BaseDataLoader = object

from k3_node.loader.sampler_utils import FastGraph


class EdgeIndex(NamedTuple):
    edge_index: Any
    e_id: Optional[Any]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        if hasattr(self.edge_index, 'to'):
            edge_index = self.edge_index.to(*args, **kwargs)
            e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None and hasattr(self.e_id, 'to') else None
            return EdgeIndex(edge_index, e_id, self.size)
        return self


class Adj(NamedTuple):
    adj_t: Any
    e_id: Optional[Any]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        if hasattr(self.adj_t, 'to'):
            adj_t = self.adj_t.to(*args, **kwargs)
            e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None and hasattr(self.e_id, 'to') else None
            return Adj(adj_t, e_id, self.size)
        return self


class NeighborSampler(BaseDataLoader):
    r"""The legacy layer-by-layer bipartite neighbor sampler from "Inductive Representation
    Learning on Large Graphs".

    Args:
        edge_index (Tensor): Edge connectivity.
        sizes (List[int]): Number of neighbors to sample per layer.
        node_idx (Tensor, optional): Seed nodes to consider. (default: :obj:`None`)
        num_nodes (int, optional): Number of nodes. (default: :obj:`None`)
        return_e_id (bool, optional): Whether to return edge IDs. (default: :obj:`True`)
        transform (callable, optional): Transform function. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        edge_index: Any,
        sizes: List[int],
        node_idx: Optional[Any] = None,
        num_nodes: Optional[int] = None,
        return_e_id: bool = True,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform

        is_torch = torch is not None and isinstance(edge_index, Tensor)
        if is_torch:
            np_edge_index = edge_index.detach().cpu().numpy()
        else:
            np_edge_index = np.asarray(edge_index)

        if num_nodes is None:
            num_nodes = int(np.max(np_edge_index) + 1) if np_edge_index.size > 0 else 0

        self.num_nodes = num_nodes
        self.graph = FastGraph(np_edge_index, num_nodes=num_nodes)

        if node_idx is None:
            node_idx = torch.arange(num_nodes) if is_torch else np.arange(num_nodes)
        elif is_torch and isinstance(node_idx, Tensor) and node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)
        elif isinstance(node_idx, np.ndarray) and node_idx.dtype == bool:
            node_idx = np.nonzero(node_idx)[0]

        self.node_idx = node_idx
        idx_list = node_idx.tolist() if hasattr(node_idx, 'tolist') else list(node_idx)

        if torch is not None:
            super().__init__(idx_list, collate_fn=self.sample, **kwargs)
        else:
            self.dataset = idx_list
            self.collate_fn = self.sample

    def sample(self, batch: Any) -> Any:
        is_torch = torch is not None
        if is_torch and not isinstance(batch, Tensor):
            batch = torch.tensor(batch, dtype=torch.long)
            batch_size = batch.numel()
            curr_nodes = batch.tolist()
        else:
            batch_size = len(batch)
            curr_nodes = list(batch)

        adjs = []
        n_id = list(curr_nodes)
        node_map = {n: i for i, n in enumerate(n_id)}

        for size in self.sizes:
            rows, cols, e_ids = [], [], []
            targets = list(curr_nodes)
            next_curr_nodes = []

            for target in targets:
                srcs, edges = self.graph.get_neighbors(target)
                count = len(srcs)
                if count == 0:
                    continue

                if size == -1 or size >= count:
                    chosen = np.arange(count)
                else:
                    chosen = np.random.choice(count, size=size, replace=False)

                for s, e in zip(srcs[chosen], edges[chosen]):
                    s = int(s)
                    e = int(e)
                    if s not in node_map:
                        node_map[s] = len(n_id)
                        n_id.append(s)
                        next_curr_nodes.append(s)

                    rows.append(node_map[s])
                    cols.append(node_map[target])
                    e_ids.append(e)

            curr_nodes = list(n_id)

            if len(rows) > 0:
                edge_index = np.stack([np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)], axis=0)
                edge_ids = np.array(e_ids, dtype=np.int64) if self.return_e_id else None
            else:
                edge_index = np.empty((2, 0), dtype=np.int64)
                edge_ids = np.empty(0, dtype=np.int64) if self.return_e_id else None

            bipartite_size = (len(n_id), len(targets))

            if is_torch:
                edge_index = torch.from_numpy(edge_index)
                if edge_ids is not None:
                    edge_ids = torch.from_numpy(edge_ids)

            adjs.append(EdgeIndex(edge_index, edge_ids, bipartite_size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out_n_id = torch.tensor(n_id, dtype=torch.long) if is_torch else np.array(n_id, dtype=np.int64)
        out = (batch_size, out_n_id, adjs)
        return self.transform(*out) if self.transform is not None else out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'
