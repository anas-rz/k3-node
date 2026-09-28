import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor
except ImportError:
    torch = None
    Tensor = type(None)


class FastGraph:
    r"""Fast CSR-based adjacency index for neighbor lookups in pure Python / NumPy."""
    def __init__(self, edge_index: Any, num_nodes: Optional[int] = None):
        if torch is not None and isinstance(edge_index, Tensor):
            np_edge_index = edge_index.detach().cpu().numpy()
        else:
            np_edge_index = np.asarray(edge_index)

        row, col = np_edge_index[0], np_edge_index[1]
        if num_nodes is None:
            num_nodes = int(max(np.max(row), np.max(col)) + 1) if len(row) > 0 else 0

        self.num_nodes = num_nodes
        self.num_edges = len(row)

        # Build CSC layout (target -> sources) for incoming neighbor sampling
        # col is target, row is source
        order = np.argsort(col, kind='mergesort')
        sorted_col = col[order]
        self.sorted_row = row[order]
        self.sorted_edge_id = order.astype(np.int64)

        # Compute indptr
        counts = np.bincount(sorted_col, minlength=num_nodes)
        self.indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        np.cumsum(counts, out=self.indptr[1:])

    def local_map(self, size: int) -> np.ndarray:
        """A reusable global-to-local node id array filled with -1 (callers must reset what they set)."""
        local = getattr(self, "_local", None)
        if local is None or len(local) < size:
            local = np.full(size, -1, dtype=np.int64)
            self._local = local
        return local

    def get_neighbors(self, node: int) -> Tuple[np.ndarray, np.ndarray]:
        if node >= self.num_nodes:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        start = self.indptr[node]
        end = self.indptr[node + 1]
        return self.sorted_row[start:end], self.sorted_edge_id[start:end]


def _csr_ranges(starts: np.ndarray, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Flattens the ranges ``[starts[i], starts[i] + counts[i])``; returns (segment id, position)."""
    counts = counts.astype(np.int64)
    segment = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    offsets = np.arange(int(counts.sum()), dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    return segment, starts.astype(np.int64)[segment] + offsets


def _sample_positions(
    starts: np.ndarray, counts: np.ndarray, k: int, replace: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Picks up to ``k`` CSR positions per segment (all of them if ``k == -1`` or ``count <= k``).

    Returns (segment id, position), grouped by segment in ascending segment order.
    """
    take_all = (counts <= k) | (k == -1)
    seg_all, pos_all = _csr_ranges(starts[take_all], counts[take_all])
    seg_all = np.nonzero(take_all)[0][seg_all]

    sampled = np.nonzero(~take_all)[0]
    if len(sampled) == 0:
        return seg_all, pos_all
    if replace:
        seg_s = np.repeat(sampled, k)
        pos_s = starts[seg_s] + np.floor(np.random.random(len(seg_s)) * counts[seg_s]).astype(np.int64)
    else:
        # Random keys per candidate; the k smallest keys of each segment form a uniform subset.
        seg_c, pos_c = _csr_ranges(starts[sampled], counts[sampled])
        order = np.lexsort((np.random.random(len(seg_c)), seg_c))
        seg_c, pos_c = seg_c[order], pos_c[order]
        first = np.searchsorted(seg_c, seg_c, side="left")
        keep = (np.arange(len(seg_c)) - first) < k
        seg_s, pos_s = sampled[seg_c[keep]], pos_c[keep]

    segment = np.concatenate([seg_all, seg_s])
    position = np.concatenate([pos_all, pos_s])
    order = np.argsort(segment, kind="stable")
    return segment[order], position[order]


def sample_neighbors_homo(
    edge_index: Any,
    seed_nodes: Any,
    num_neighbors: List[int],
    num_nodes: Optional[int] = None,
    replace: bool = False,
    subgraph_type: str = 'directional',
    disjoint: bool = False,
    graph: Optional[FastGraph] = None,
) -> Tuple[Any, Any, Any, Any, List[int], List[int]]:
    r"""Samples multi-hop neighborhoods on a homogeneous graph.

    All steps are vectorized with NumPy. Pass a prebuilt ``graph`` (a :class:`FastGraph` of
    ``edge_index``) to avoid rebuilding the CSR index for every batch.

    Returns:
        (sampled_nodes, local_row, local_col, edge_ids, num_sampled_nodes, num_sampled_edges)
    """
    is_torch = torch is not None and isinstance(seed_nodes, Tensor)
    if is_torch:
        device = seed_nodes.device
        np_seeds = seed_nodes.detach().cpu().numpy()
    else:
        device = None
        np_seeds = np.asarray(seed_nodes)
    np_seeds = np_seeds.astype(np.int64).reshape(-1)

    if graph is None:
        graph = FastGraph(edge_index, num_nodes=num_nodes)

    # Global -> local id map; only the entries touched here are reset afterwards.
    size = max(graph.num_nodes, int(np_seeds.max()) + 1 if len(np_seeds) else 0)
    local = graph.local_map(size)

    _, first_idx = np.unique(np_seeds, return_index=True)
    first_idx = np.sort(first_idx)
    frontier = np_seeds[first_idx]
    node_chunks = [frontier]
    local[frontier] = np.arange(len(frontier))
    num_total = len(frontier)
    batch_ids = first_idx.astype(np.int64) if disjoint else None

    num_sampled_nodes = [len(frontier)]
    num_sampled_edges = []
    hop_src, hop_dst, hop_eid = [], [], []
    try:
        for k in num_neighbors:
            in_graph = frontier < graph.num_nodes
            starts = np.zeros(len(frontier), dtype=np.int64)
            counts = np.zeros(len(frontier), dtype=np.int64)
            starts[in_graph] = graph.indptr[frontier[in_graph]]
            counts[in_graph] = graph.indptr[frontier[in_graph] + 1] - starts[in_graph]

            segment, position = _sample_positions(starts, counts, k, replace)
            srcs = graph.sorted_row[position].astype(np.int64)
            dsts = frontier[segment]
            hop_src.append(srcs)
            hop_dst.append(dsts)
            hop_eid.append(graph.sorted_edge_id[position])
            num_sampled_edges.append(len(srcs))

            # Unvisited sources become new nodes, in order of first appearance.
            is_new = local[srcs] == -1
            new_srcs = srcs[is_new]
            uniq, first = np.unique(new_srcs, return_index=True)
            order = np.argsort(first, kind="stable")
            new_nodes = uniq[order]
            local[new_nodes] = num_total + np.arange(len(new_nodes))
            if disjoint:
                discoverer = dsts[is_new][first[order]]
                batch_ids = np.concatenate([batch_ids, batch_ids[local[discoverer]]])
            num_total += len(new_nodes)
            node_chunks.append(new_nodes)
            num_sampled_nodes.append(len(new_nodes))
            frontier = new_nodes

        nodes = np.concatenate(node_chunks).astype(np.int64)

        if subgraph_type == 'induced':
            # Every original edge whose endpoints were both visited, in original edge order.
            targets = nodes[nodes < graph.num_nodes]
            t_starts = graph.indptr[targets]
            segment, position = _csr_ranges(t_starts, graph.indptr[targets + 1] - t_starts)
            srcs = graph.sorted_row[position].astype(np.int64)
            keep = local[srcs] != -1
            edge_ids = graph.sorted_edge_id[position][keep]
            order = np.argsort(edge_ids, kind="stable")
            edge_ids = edge_ids[order]
            local_row = local[srcs[keep]][order]
            local_col = local[targets[segment[keep]]][order]
        else:
            srcs = np.concatenate(hop_src) if hop_src else np.empty(0, dtype=np.int64)
            dsts = np.concatenate(hop_dst) if hop_dst else np.empty(0, dtype=np.int64)
            edge_ids = np.concatenate(hop_eid) if hop_eid else np.empty(0, dtype=np.int64)
            local_row, local_col = local[srcs], local[dsts]
            if subgraph_type == 'bidirectional':
                # Each sampled edge followed by its reverse.
                local_row, local_col = (
                    np.stack([local_row, local_col], axis=1).reshape(-1),
                    np.stack([local_col, local_row], axis=1).reshape(-1),
                )
                edge_ids = np.repeat(edge_ids, 2)
    finally:
        local[np.concatenate(node_chunks)] = -1

    local_row = local_row.astype(np.int64)
    local_col = local_col.astype(np.int64)
    edge_ids = edge_ids.astype(np.int64)

    if is_torch:
        nodes = torch.from_numpy(nodes).to(device=device, dtype=torch.long)
        local_row = torch.from_numpy(local_row).to(device=device, dtype=torch.long)
        local_col = torch.from_numpy(local_col).to(device=device, dtype=torch.long)
        edge_ids = torch.from_numpy(edge_ids).to(device=device, dtype=torch.long)

    return nodes, local_row, local_col, edge_ids, num_sampled_nodes, num_sampled_edges


def sample_neighbors_hetero(
    edge_index_dict: Dict[Tuple[str, str, str], Any],
    seed_nodes_dict: Dict[str, Any],
    num_neighbors: Union[List[int], Dict[Tuple[str, str, str], List[int]]],
    num_nodes_dict: Optional[Dict[str, int]] = None,
    replace: bool = False,
    subgraph_type: str = 'directional',
) -> Tuple[Dict[str, Any], Dict[Tuple[str, str, str], Any], Dict[Tuple[str, str, str], Any], Dict[Tuple[str, str, str], Any], Dict[str, List[int]], Dict[Tuple[str, str, str], List[int]]]:
    r"""Samples multi-hop neighborhoods on a heterogeneous graph."""
    # Build fast graphs per edge type
    graphs = {}
    for edge_type, edge_index in edge_index_dict.items():
        graphs[edge_type] = FastGraph(edge_index)

    # Determine num_hops
    if isinstance(num_neighbors, dict):
        first_key = list(num_neighbors.keys())[0]
        num_hops = len(num_neighbors[first_key])
    else:
        num_hops = len(num_neighbors)

    is_torch = torch is not None and any(isinstance(v, Tensor) for v in seed_nodes_dict.values())
    device = None
    if is_torch:
        for v in seed_nodes_dict.values():
            if isinstance(v, Tensor):
                device = v.device
                break

    nodes_dict = {k: [] for k in seed_nodes_dict.keys()}
    visited_dict = {k: {} for k in seed_nodes_dict.keys()}

    for node_type, seeds in seed_nodes_dict.items():
        if seeds is None:
            continue
        np_seeds = seeds.detach().cpu().numpy() if (torch is not None and isinstance(seeds, Tensor)) else np.asarray(seeds)
        for s in np_seeds:
            s = int(s)
            if s not in visited_dict[node_type]:
                visited_dict[node_type][s] = len(nodes_dict[node_type])
                nodes_dict[node_type].append(s)

    frontier_dict = {k: list(v) for k, v in nodes_dict.items()}
    num_sampled_nodes = {k: [len(v)] for k, v in nodes_dict.items()}
    num_sampled_edges = {k: [] for k in edge_index_dict.keys()}
    sampled_edges_dict = {k: [] for k in edge_index_dict.keys()}

    for hop in range(num_hops):
        next_frontier_dict = {k: [] for k in visited_dict.keys()}

        for edge_type, graph in graphs.items():
            src_type, rel, dst_type = edge_type
            if dst_type not in frontier_dict:
                continue

            if isinstance(num_neighbors, dict):
                k = num_neighbors[edge_type][hop]
            else:
                k = num_neighbors[hop]

            hop_edges = 0
            for target in frontier_dict[dst_type]:
                srcs, e_ids = graph.get_neighbors(target)
                count = len(srcs)
                if count == 0:
                    continue

                if k == -1 or k >= count:
                    chosen_idx = np.arange(count)
                else:
                    chosen_idx = np.random.choice(count, size=k, replace=replace)

                for s, e in zip(srcs[chosen_idx], e_ids[chosen_idx]):
                    s = int(s)
                    e = int(e)
                    sampled_edges_dict[edge_type].append((s, target, e))
                    hop_edges += 1

                    if src_type not in visited_dict:
                        visited_dict[src_type] = {}
                        nodes_dict[src_type] = []
                        next_frontier_dict[src_type] = []
                        num_sampled_nodes[src_type] = [0] * (hop + 1)

                    if s not in visited_dict[src_type]:
                        visited_dict[src_type][s] = len(nodes_dict[src_type])
                        nodes_dict[src_type].append(s)
                        next_frontier_dict[src_type].append(s)

            num_sampled_edges[edge_type].append(hop_edges)

        frontier_dict = next_frontier_dict
        for k in nodes_dict.keys():
            count = len(next_frontier_dict.get(k, []))
            if k in num_sampled_nodes:
                num_sampled_nodes[k].append(count)

    # Build local rows, cols, edge_ids
    out_row = {}
    out_col = {}
    out_edge = {}

    for edge_type, edge_list in sampled_edges_dict.items():
        src_type, rel, dst_type = edge_type
        rows, cols, eids = [], [], []
        for u, v, e in edge_list:
            rows.append(visited_dict[src_type][u])
            cols.append(visited_dict[dst_type][v])
            eids.append(e)

        if len(rows) > 0:
            r = np.array(rows, dtype=np.int64)
            c = np.array(cols, dtype=np.int64)
            e = np.array(eids, dtype=np.int64)
        else:
            r = np.empty(0, dtype=np.int64)
            c = np.empty(0, dtype=np.int64)
            e = np.empty(0, dtype=np.int64)

        if is_torch:
            out_row[edge_type] = torch.from_numpy(r).to(device=device, dtype=torch.long)
            out_col[edge_type] = torch.from_numpy(c).to(device=device, dtype=torch.long)
            out_edge[edge_type] = torch.from_numpy(e).to(device=device, dtype=torch.long)
        else:
            out_row[edge_type] = r
            out_col[edge_type] = c
            out_edge[edge_type] = e

    out_nodes = {}
    for k, v in nodes_dict.items():
        arr = np.array(v, dtype=np.int64)
        if is_torch:
            out_nodes[k] = torch.from_numpy(arr).to(device=device, dtype=torch.long)
        else:
            out_nodes[k] = arr

    return out_nodes, out_row, out_col, out_edge, num_sampled_nodes, num_sampled_edges


def random_walk(edge_index: Any, start_nodes: Any, walk_length: int, num_nodes: Optional[int] = None) -> Any:
    r"""Executes random walks from start_nodes."""
    graph = FastGraph(edge_index, num_nodes=num_nodes)
    is_torch = torch is not None and isinstance(start_nodes, Tensor)
    np_starts = start_nodes.detach().cpu().numpy() if is_torch else np.asarray(start_nodes)

    walks = []
    for s in np_starts:
        curr = int(s)
        walk = [curr]
        for _ in range(walk_length):
            srcs, _ = graph.get_neighbors(curr)
            if len(srcs) == 0:
                walk.append(curr)
            else:
                curr = int(np.random.choice(srcs))
                walk.append(curr)
        walks.append(walk)

    walks_arr = np.array(walks, dtype=np.int64)
    if is_torch:
        return torch.from_numpy(walks_arr).to(device=start_nodes.device, dtype=torch.long)
    return walks_arr


def partition_graph(edge_index: Any, num_nodes: int, num_parts: int) -> Any:
    r"""Partitions graph nodes into num_parts clusters."""
    if num_parts <= 1:
        cluster = np.zeros(num_nodes, dtype=np.int64)
        if torch is not None and isinstance(edge_index, Tensor):
            return torch.from_numpy(cluster).to(device=edge_index.device, dtype=torch.long)
        return cluster

    # Try Metis if installed
    try:
        import torch_geometric.typing as pyg_typing
        if hasattr(pyg_typing, 'WITH_TORCH_SPARSE') and pyg_typing.WITH_TORCH_SPARSE:
            from torch_geometric.index import index2ptr
            from torch_geometric.utils import sort_edge_index
            row, col = sort_edge_index(edge_index, num_nodes=num_nodes)
            indptr = index2ptr(row, size=num_nodes)
            return torch.ops.torch_sparse.partition(indptr.cpu(), col.cpu(), None, num_parts, False).to(edge_index.device)
    except Exception:
        pass

    # Fast BFS / linear partition fallback
    cluster = np.full(num_nodes, -1, dtype=np.int64)
    part_size = math.ceil(num_nodes / num_parts)

    graph = FastGraph(edge_index, num_nodes=num_nodes)
    unassigned = set(range(num_nodes))

    current_part = 0
    while unassigned and current_part < num_parts:
        seed = next(iter(unassigned))
        queue = [seed]
        unassigned.remove(seed)
        cluster[seed] = current_part
        assigned_in_part = 1

        while queue and assigned_in_part < part_size:
            v = queue.pop(0)
            srcs, _ = graph.get_neighbors(v)
            for u in srcs:
                u = int(u)
                if u in unassigned:
                    unassigned.remove(u)
                    cluster[u] = current_part
                    queue.append(u)
                    assigned_in_part += 1
                    if assigned_in_part >= part_size:
                        break

        current_part += 1

    # Any remaining nodes get distributed
    if unassigned:
        for idx, u in enumerate(unassigned):
            cluster[u] = idx % num_parts

    if torch is not None and isinstance(edge_index, Tensor):
        return torch.from_numpy(cluster).to(device=edge_index.device, dtype=torch.long)
    return cluster



def sample_neighbors_disjoint(graph: FastGraph, seed_nodes: Any, num_neighbors: List[int], replace: bool = False):
    r"""Samples a separate multi-hop neighborhood for every seed node, as PyG's ``disjoint=True``.

    A node reached from two seeds appears twice, once in each seed's subgraph. The sampled edges
    point from sources to the nodes that sampled them ("directional").

    Returns:
        (nodes, local_row, local_col, edge_ids, batch, num_sampled_nodes, num_sampled_edges), where
        ``batch[i]`` is the seed (subgraph) that local node ``i`` belongs to.
    """
    seeds = np.asarray(seed_nodes).astype(np.int64).reshape(-1)
    num_nodes = graph.num_nodes
    frontier_nodes, frontier_ids = seeds, np.arange(len(seeds), dtype=np.int64)
    node_chunks, batch_chunks = [seeds], [np.arange(len(seeds), dtype=np.int64)]
    keys = np.arange(len(seeds), dtype=np.int64) * num_nodes + seeds  # (seed, node) of every local node
    sorted_order = np.argsort(keys)
    num_total = len(seeds)
    rows, cols, eids = [], [], []
    num_sampled_nodes, num_sampled_edges = [len(seeds)], []
    for k in num_neighbors:
        in_graph = frontier_nodes < num_nodes
        starts = np.zeros(len(frontier_nodes), dtype=np.int64)
        counts = np.zeros(len(frontier_nodes), dtype=np.int64)
        starts[in_graph] = graph.indptr[frontier_nodes[in_graph]]
        counts[in_graph] = graph.indptr[frontier_nodes[in_graph] + 1] - starts[in_graph]
        segment, position = _sample_positions(starts, counts, k, replace)
        srcs = graph.sorted_row[position].astype(np.int64)
        dst_ids = frontier_ids[segment]
        src_batch = np.concatenate(batch_chunks)[dst_ids]
        src_keys = src_batch * num_nodes + srcs

        # Look up which (seed, node) pairs already exist; the rest become new local nodes
        pos = np.searchsorted(keys[sorted_order], src_keys)
        pos = np.minimum(pos, len(keys) - 1)
        known = keys[sorted_order][pos] == src_keys
        src_ids = np.where(known, sorted_order[pos], -1)
        new_keys, first, inverse = np.unique(src_keys[~known], return_index=True, return_inverse=True)
        order = np.argsort(first, kind="stable")  # new nodes in order of first appearance
        rank = np.empty(len(order), dtype=np.int64)
        rank[order] = np.arange(len(order))
        src_ids[~known] = num_total + rank[inverse.reshape(-1)]
        new_keys = new_keys[order]

        rows.append(src_ids)
        cols.append(dst_ids)
        eids.append(graph.sorted_edge_id[position])
        num_sampled_edges.append(len(srcs))
        new_nodes, new_batch = new_keys % num_nodes, new_keys // num_nodes
        node_chunks.append(new_nodes)
        batch_chunks.append(new_batch)
        keys = np.concatenate([keys, new_keys])
        sorted_order = np.argsort(keys, kind="stable")
        frontier_nodes, frontier_ids = new_nodes, num_total + np.arange(len(new_nodes))
        num_total += len(new_nodes)
        num_sampled_nodes.append(len(new_nodes))

    empty = np.empty(0, dtype=np.int64)
    return (np.concatenate(node_chunks), np.concatenate(rows) if rows else empty,
            np.concatenate(cols) if cols else empty, np.concatenate(eids).astype(np.int64) if eids else empty,
            np.concatenate(batch_chunks), num_sampled_nodes, num_sampled_edges)
