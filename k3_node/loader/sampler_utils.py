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

    def get_neighbors(self, node: int) -> Tuple[np.ndarray, np.ndarray]:
        if node >= self.num_nodes:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        start = self.indptr[node]
        end = self.indptr[node + 1]
        return self.sorted_row[start:end], self.sorted_edge_id[start:end]


def sample_neighbors_homo(
    edge_index: Any,
    seed_nodes: Any,
    num_neighbors: List[int],
    num_nodes: Optional[int] = None,
    replace: bool = False,
    subgraph_type: str = 'directional',
    disjoint: bool = False,
) -> Tuple[Any, Any, Any, Any, List[int], List[int]]:
    r"""Samples multi-hop neighborhoods on a homogeneous graph.

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

    graph = FastGraph(edge_index, num_nodes=num_nodes)

    # Initialize sampling state
    nodes = []
    visited = {}  # global_node -> local_id
    batch_ids = []  # for disjoint sampling

    for i, s in enumerate(np_seeds):
        s = int(s)
        if s not in visited:
            visited[s] = len(nodes)
            nodes.append(s)
            if disjoint:
                batch_ids.append(i)

    frontier = list(nodes)
    num_sampled_nodes = [len(nodes)]
    num_sampled_edges = []
    sampled_edges_list = []  # (src, dst, edge_id)

    for hop, k in enumerate(num_neighbors):
        next_frontier = []
        hop_edges = 0

        for target in frontier:
            srcs, e_ids = graph.get_neighbors(target)
            count = len(srcs)
            if count == 0:
                continue

            if k == -1 or k >= count:
                chosen_idx = np.arange(count)
            else:
                chosen_idx = np.random.choice(count, size=k, replace=replace)

            chosen_srcs = srcs[chosen_idx]
            chosen_e_ids = e_ids[chosen_idx]

            for s, e in zip(chosen_srcs, chosen_e_ids):
                s = int(s)
                e = int(e)
                sampled_edges_list.append((s, target, e))
                hop_edges += 1

                if s not in visited:
                    visited[s] = len(nodes)
                    nodes.append(s)
                    next_frontier.append(s)
                    if disjoint:
                        batch_ids.append(batch_ids[visited[target]])

        frontier = next_frontier
        num_sampled_nodes.append(len(next_frontier))
        num_sampled_edges.append(hop_edges)

    # Resolve subgraph type
    if subgraph_type == 'induced':
        # Include all edges in original graph where both endpoints are in visited
        local_edges = []
        edge_ids = []
        if is_torch:
            np_edge_index = edge_index.detach().cpu().numpy()
        else:
            np_edge_index = np.asarray(edge_index)
        row_all, col_all = np_edge_index[0], np_edge_index[1]
        for e_idx, (u, v) in enumerate(zip(row_all, col_all)):
            u, v = int(u), int(v)
            if u in visited and v in visited:
                local_edges.append((visited[u], visited[v]))
                edge_ids.append(e_idx)
    else:
        local_edges = []
        edge_ids = []
        for u, v, e in sampled_edges_list:
            local_edges.append((visited[u], visited[v]))
            edge_ids.append(e)
            if subgraph_type == 'bidirectional':
                local_edges.append((visited[v], visited[u]))
                edge_ids.append(e)

    if len(local_edges) > 0:
        local_row = np.array([e[0] for e in local_edges], dtype=np.int64)
        local_col = np.array([e[1] for e in local_edges], dtype=np.int64)
        edge_ids = np.array(edge_ids, dtype=np.int64)
    else:
        local_row = np.empty(0, dtype=np.int64)
        local_col = np.empty(0, dtype=np.int64)
        edge_ids = np.empty(0, dtype=np.int64)

    nodes = np.array(nodes, dtype=np.int64)

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

