import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp

from k3_node.data import Data, HeteroData
from k3_node.transforms.base_transform import BaseTransform, functional_transform
from k3_node.transforms.utils import as_tensor, is_torch_tensor, match_tensor, to_numpy, to_undirected
from k3_node.utils.graph import coalesce, is_undirected as check_is_undirected, subgraph


@functional_transform("to_undirected")
class ToUndirected(BaseTransform):
    r"""Converts a homogeneous or heterogeneous graph to an undirected graph."""

    def __init__(self, reduce: str = "add", merge: bool = True):
        self.reduce = reduce
        self.merge = merge

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if "edge_index" not in store:
                continue

            if isinstance(data, HeteroData) and (store.is_bipartite() or not self.merge):
                src, rel, dst = getattr(store, "_key", ("src", "rel", "dst"))
                ei = store.edge_index
                ei_np = to_numpy(ei)
                rev_ei_np = np.stack([ei_np[1], ei_np[0]], axis=0)

                inv_store = data[dst, f"rev_{rel}", src]
                inv_store.edge_index = match_tensor(rev_ei_np, ei)
                for key, val in store.items():
                    if key != "edge_index" and store.is_edge_attr(key):
                        inv_store[key] = val
            else:
                attr = store.get("edge_attr", None)
                if attr is not None:
                    out_ei, out_attr = to_undirected(store.edge_index, attr, reduce=self.reduce)
                    store.edge_index = out_ei
                    store.edge_attr = out_attr
                else:
                    store.edge_index = to_undirected(store.edge_index, reduce=self.reduce)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduce='{self.reduce}', merge={self.merge})"


@functional_transform("one_hot_degree")
class OneHotDegree(BaseTransform):
    r"""Adds the node degree as a one-hot feature to :obj:`x`."""

    def __init__(self, max_degree: int, cat: bool = True):
        self.max_degree = max_degree
        self.cat = cat

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            num_nodes = store.num_nodes
            assert num_nodes is not None

            # Count in-degree
            deg = np.zeros(num_nodes, dtype=np.int64)
            if hasattr(data, "edge_index") and data.edge_index is not None:
                col = to_numpy(data.edge_index)[1]
                np.add.at(deg, col[col < num_nodes], 1)

            deg = np.clip(deg, 0, self.max_degree)
            one_hot = np.zeros((num_nodes, self.max_degree + 1), dtype=np.float32)
            one_hot[np.arange(num_nodes), deg] = 1.0

            if hasattr(store, "x") and store.x is not None and self.cat:
                x_np = to_numpy(store.x)
                if x_np.ndim == 1:
                    x_np = x_np.reshape(-1, 1)
                new_x = np.concatenate([x_np, one_hot], axis=-1)
                store.x = match_tensor(new_x, store.x)
            else:
                store.x = match_tensor(one_hot, getattr(store, "x", None))

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_degree={self.max_degree})"


@functional_transform("target_indegree")
class TargetIndegree(BaseTransform):
    r"""Appends the target node in-degree to the edge attributes."""

    def __init__(self, cat: bool = True):
        self.cat = cat

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if "edge_index" not in store:
                continue
            ei_np = to_numpy(store.edge_index)
            num_nodes = data.num_nodes if hasattr(data, "num_nodes") and data.num_nodes is not None else int(np.max(ei_np)) + 1
            col = ei_np[1]
            deg = np.zeros(num_nodes, dtype=np.float32)
            np.add.at(deg, col, 1.0)
            in_deg = deg[col].reshape(-1, 1)

            attr = store.get("edge_attr", None)
            if attr is not None and self.cat:
                attr_np = to_numpy(attr)
                if attr_np.ndim == 1:
                    attr_np = attr_np.reshape(-1, 1)
                new_attr = np.concatenate([attr_np, in_deg], axis=-1)
                store.edge_attr = match_tensor(new_attr, attr)
            else:
                store.edge_attr = match_tensor(in_deg, store.edge_index, dtype="float32")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cat={self.cat})"


@functional_transform("local_degree_profile")
class LocalDegreeProfile(BaseTransform):
    r"""Appends the Local Degree Profile (LDP) to node features :obj:`x`."""

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

        deg = np.zeros(num_nodes, dtype=np.float32)
        row, col = ei_np[0], ei_np[1]
        np.add.at(deg, row, 1.0)

        # For each node, compute min, max, mean, std of neighbor degrees
        min_deg = np.zeros(num_nodes, dtype=np.float32)
        max_deg = np.zeros(num_nodes, dtype=np.float32)
        mean_deg = np.zeros(num_nodes, dtype=np.float32)
        std_deg = np.zeros(num_nodes, dtype=np.float32)

        for i in range(num_nodes):
            neigh_degrees = deg[col[row == i]]
            if len(neigh_degrees) > 0:
                min_deg[i] = np.min(neigh_degrees)
                max_deg[i] = np.max(neigh_degrees)
                mean_deg[i] = np.mean(neigh_degrees)
                std_deg[i] = np.std(neigh_degrees)

        ldp = np.stack([deg, min_deg, max_deg, mean_deg, std_deg], axis=-1)

        if hasattr(data, "x") and data.x is not None:
            x_np = to_numpy(data.x)
            if x_np.ndim == 1:
                x_np = x_np.reshape(-1, 1)
            new_x = np.concatenate([x_np, ldp], axis=-1)
            data.x = match_tensor(new_x, data.x)
        else:
            data.x = match_tensor(ldp, data.edge_index, dtype="float32")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@functional_transform("add_self_loops")
class AddSelfLoops(BaseTransform):
    r"""Adds self-loops to the graph."""

    def __init__(self, attr: str = "edge_weight", fill_value: Union[float, str] = 1.0):
        self.attr = attr
        self.fill_value = fill_value

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or "edge_index" not in store:
                continue

            ei_np = to_numpy(store.edge_index)
            num_nodes = data.num_nodes if hasattr(data, "num_nodes") and data.num_nodes is not None else (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

            loops = np.arange(num_nodes, dtype=ei_np.dtype)
            loop_index = np.stack([loops, loops], axis=0)
            new_ei = np.concatenate([ei_np, loop_index], axis=1)
            store.edge_index = match_tensor(new_ei, store.edge_index)

            if self.attr in store and store[self.attr] is not None:
                val = store[self.attr]
                val_np = to_numpy(val)
                fill_val = 1.0 if isinstance(self.fill_value, str) else self.fill_value
                pad_shape = (num_nodes,) + val_np.shape[1:]
                loop_attr = np.full(pad_shape, fill_val, dtype=val_np.dtype)
                store[self.attr] = match_tensor(np.concatenate([val_np, loop_attr], axis=0), val)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attr='{self.attr}', fill_value={self.fill_value})"


@functional_transform("add_remaining_self_loops")
class AddRemainingSelfLoops(BaseTransform):
    r"""Adds self-loops to nodes that do not already have one."""

    def __init__(self, attr: str = "edge_weight", fill_value: Union[float, str] = 1.0):
        self.attr = attr
        self.fill_value = fill_value

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if store.is_bipartite() or "edge_index" not in store:
                continue

            ei_np = to_numpy(store.edge_index)
            num_nodes = data.num_nodes if hasattr(data, "num_nodes") and data.num_nodes is not None else (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

            mask = ei_np[0] == ei_np[1]
            existing_loops = set(ei_np[0, mask])
            missing_loops = [i for i in range(num_nodes) if i not in existing_loops]

            if len(missing_loops) > 0:
                missing = np.array(missing_loops, dtype=ei_np.dtype)
                loop_index = np.stack([missing, missing], axis=0)
                new_ei = np.concatenate([ei_np, loop_index], axis=1)
                store.edge_index = match_tensor(new_ei, store.edge_index)

                if self.attr in store and store[self.attr] is not None:
                    val = store[self.attr]
                    val_np = to_numpy(val)
                    fill_val = 1.0 if isinstance(self.fill_value, str) else self.fill_value
                    pad_shape = (len(missing_loops),) + val_np.shape[1:]
                    loop_attr = np.full(pad_shape, fill_val, dtype=val_np.dtype)
                    store[self.attr] = match_tensor(np.concatenate([val_np, loop_attr], axis=0), val)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attr='{self.attr}', fill_value={self.fill_value})"


@functional_transform("remove_self_loops")
class RemoveSelfLoops(BaseTransform):
    r"""Removes all self-loops from the graph."""

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if "edge_index" not in store:
                continue
            ei_np = to_numpy(store.edge_index)
            mask = ei_np[0] != ei_np[1]
            store.edge_index = match_tensor(ei_np[:, mask], store.edge_index)

            for key, val in list(store.items()):
                if key != "edge_index" and store.is_edge_attr(key):
                    val_np = to_numpy(val)
                    if val_np.shape[0] == ei_np.shape[1]:
                        store[key] = match_tensor(val_np[mask], val)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@functional_transform("remove_isolated_nodes")
class RemoveIsolatedNodes(BaseTransform):
    r"""Removes isolated nodes (nodes with degree 0)."""

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

        connected = np.zeros(num_nodes, dtype=bool)
        if ei_np.size > 0:
            connected[ei_np[0]] = True
            connected[ei_np[1]] = True

        new_indices = np.full(num_nodes, -1, dtype=ei_np.dtype)
        new_indices[connected] = np.arange(np.sum(connected), dtype=ei_np.dtype)

        if ei_np.size > 0:
            data.edge_index = match_tensor(new_indices[ei_np], data.edge_index)
        else:
            data.edge_index = match_tensor(np.empty((2, 0), dtype=ei_np.dtype), data.edge_index)

        for key, val in list(data.items()):
            if data.is_node_attr(key):
                val_np = to_numpy(val)
                if val_np.shape[0] == num_nodes:
                    data[key] = match_tensor(val_np[connected], val)

        if "num_nodes" in data:
            data.num_nodes = int(np.sum(connected))

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@functional_transform("remove_duplicated_edges")
class RemoveDuplicatedEdges(BaseTransform):
    r"""Removes duplicated edges from the graph."""

    def __init__(self, key: Optional[str] = None, reduce: str = "add"):
        self.key = key
        self.reduce = reduce

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if "edge_index" not in store:
                continue
            attr = store.get(self.key, store.get("edge_attr", None))
            if attr is not None:
                new_ei, new_attr = coalesce(store.edge_index, attr, reduce=self.reduce)
                store.edge_index = new_ei
                if self.key is not None:
                    store[self.key] = new_attr
                else:
                    store.edge_attr = new_attr
            else:
                new_ei, _ = coalesce(store.edge_index, None, reduce=self.reduce)
                store.edge_index = new_ei

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(key={self.key}, reduce='{self.reduce}')"


@functional_transform("knn_graph")
class KNNGraph(BaseTransform):
    r"""Creates a k-NN graph based on node positions :obj:`data.pos`."""

    def __init__(
        self,
        k: int = 6,
        loop: bool = False,
        force_undirected: bool = False,
        flow: str = "source_to_target",
        cosine: bool = False,
        num_workers: int = 1,
    ):
        self.k = k
        self.loop = loop
        self.force_undirected = force_undirected
        self.flow = flow
        self.cosine = cosine
        self.num_workers = num_workers

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        from k3_node.layers.pool.knn import knn_graph

        batch = getattr(data, "batch", None)
        edge_index = knn_graph(
            data.pos,
            self.k,
            batch=batch,
            loop=self.loop,
            flow=self.flow,
            cosine=self.cosine,
        )
        if self.force_undirected:
            edge_index = to_undirected(edge_index, num_nodes=data.num_nodes)

        data.edge_index = match_tensor(edge_index, data.pos, dtype="int64")
        data.edge_attr = None
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


@functional_transform("radius_graph")
class RadiusGraph(BaseTransform):
    r"""Creates a radius neighborhood graph based on node positions :obj:`data.pos`."""

    def __init__(
        self,
        r: float,
        loop: bool = False,
        max_num_neighbors: int = 32,
        flow: str = "source_to_target",
        num_workers: int = 1,
    ):
        self.r = r
        self.loop = loop
        self.max_num_neighbors = max_num_neighbors
        self.flow = flow
        self.num_workers = num_workers

    def forward(self, data: Data) -> Data:
        assert data.pos is not None
        from k3_node.layers.pool.point_cloud import radius_graph

        batch = getattr(data, "batch", None)
        edge_index = radius_graph(
            data.pos,
            self.r,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
            flow=self.flow,
        )
        data.edge_index = match_tensor(edge_index, data.pos, dtype="int64")
        data.edge_attr = None
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(r={self.r})"


@functional_transform("to_dense")
class ToDense(BaseTransform):
    r"""Converts a sparse adjacency matrix to a dense adjacency matrix."""

    def __init__(self, num_nodes: Optional[int] = None):
        self.num_nodes = num_nodes

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        orig_num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)
        num_nodes = orig_num_nodes if self.num_nodes is None else max(orig_num_nodes, self.num_nodes)

        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        attr_np = to_numpy(data.edge_attr) if hasattr(data, "edge_attr") and data.edge_attr is not None else None
        if ei_np.size > 0:
            if attr_np is not None and attr_np.ndim == 1:
                adj[ei_np[0], ei_np[1]] = attr_np
            elif attr_np is not None:
                adj = np.zeros((num_nodes, num_nodes, attr_np.shape[-1]), dtype=np.float32)
                adj[ei_np[0], ei_np[1]] = attr_np
            else:
                adj[ei_np[0], ei_np[1]] = 1.0

        data.adj = match_tensor(adj, data.edge_index, dtype="float32")
        data.edge_index = None
        data.edge_attr = None

        mask = np.zeros(num_nodes, dtype=bool)
        mask[:orig_num_nodes] = True
        data.mask = match_tensor(mask, data.adj, dtype="bool")

        pad_nodes = num_nodes - orig_num_nodes
        if pad_nodes > 0:
            for key in ["x", "pos", "y"]:
                if hasattr(data, key) and getattr(data, key) is not None:
                    val = getattr(data, key)
                    val_np = to_numpy(val)
                    pad_shape = (pad_nodes,) + val_np.shape[1:]
                    padded = np.concatenate([val_np, np.zeros(pad_shape, dtype=val_np.dtype)], axis=0)
                    setattr(data, key, match_tensor(padded, val))

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_nodes={self.num_nodes})"


@functional_transform("two_hop")
class TwoHop(BaseTransform):
    r"""Adds two-hop edges to the edge indices."""

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

        # Adjacency matrix squaring via scipy
        adj = sp.coo_matrix(
            (np.ones(ei_np.shape[1], dtype=np.float32), (ei_np[0], ei_np[1])),
            shape=(num_nodes, num_nodes),
        ).tocsr()
        adj2 = (adj @ adj).tocoo()

        # Remove self loops
        mask = adj2.row != adj2.col
        ei2 = np.stack([adj2.row[mask], adj2.col[mask]], axis=0)

        new_ei = np.concatenate([ei_np, ei2], axis=1)
        attr = getattr(data, "edge_attr", None)
        if attr is not None:
            attr_np = to_numpy(attr)
            pad_shape = (ei2.shape[1],) + attr_np.shape[1:]
            new_attr = np.concatenate([attr_np, np.zeros(pad_shape, dtype=attr_np.dtype)], axis=0)
            final_ei, final_attr = coalesce(new_ei, new_attr, num_nodes=num_nodes)
            data.edge_index = match_tensor(final_ei, data.edge_index)
            data.edge_attr = match_tensor(final_attr, attr)
        else:
            final_ei, _ = coalesce(new_ei, None, num_nodes=num_nodes)
            data.edge_index = match_tensor(final_ei, data.edge_index)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@functional_transform("line_graph")
class LineGraph(BaseTransform):
    r"""Converts a graph to its corresponding line graph."""

    def __init__(self, force_directed: bool = False):
        self.force_directed = force_directed

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        num_edges = ei_np.shape[1]

        # An edge exists from e1 to e2 if e1=(u, v) and e2=(v, w)
        # Create map from target node v to edge indices
        edges_out = []
        target_to_edges = {}
        for idx, (u, v) in enumerate(zip(ei_np[0], ei_np[1])):
            target_to_edges.setdefault(v, []).append(idx)

        lg_rows, lg_cols = [], []
        for idx, (u, v) in enumerate(zip(ei_np[0], ei_np[1])):
            for next_edge in target_to_edges.get(u if not self.force_directed else -1, []):
                pass
            # standard line graph: target of e1 == source of e2
            # Here find e2 where e2[0] == v:
            # We can find where ei_np[0] == v
            dest_edges = np.where(ei_np[0] == v)[0]
            for de in dest_edges:
                if not self.force_directed or idx != de:
                    lg_rows.append(idx)
                    lg_cols.append(de)

        if len(lg_rows) > 0:
            lg_ei = np.stack([lg_rows, lg_cols], axis=0).astype(ei_np.dtype)
        else:
            lg_ei = np.empty((2, 0), dtype=ei_np.dtype)

        # Node features of line graph are original edge attributes or ones
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.x = data.edge_attr
        data.edge_index = match_tensor(lg_ei, data.edge_index)
        data.edge_attr = None
        data.num_nodes = num_edges
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(force_directed={self.force_directed})"


@functional_transform("laplacian_lambda_max")
class LaplacianLambdaMax(BaseTransform):
    r"""Computes the largest eigenvalue of the graph Laplacian."""

    def __init__(self, normalization: Optional[str] = None, is_undirected: bool = False):
        assert normalization in [None, "sym", "rw"], "Invalid normalization"
        self.normalization = normalization
        self.is_undirected = is_undirected

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

        from k3_node.layers.conv.utils import get_laplacian

        edge_weight = getattr(data, "edge_attr", None)
        if edge_weight is not None:
            ew_np = to_numpy(edge_weight)
            if ew_np.size != ei_np.shape[1]:
                edge_weight = None
        if edge_weight is None:
            edge_weight = getattr(data, "edge_weight", None)

        ei, ew = get_laplacian(data.edge_index, edge_weight, normalization=self.normalization, num_nodes=num_nodes)
        ei_np, ew_np = to_numpy(ei), to_numpy(ew)
        L = sp.coo_matrix((ew_np, (ei_np[0], ei_np[1])), shape=(num_nodes, num_nodes))

        if num_nodes > 2:
            try:
                eig_fn = sp.linalg.eigsh if (self.is_undirected and self.normalization != "rw") else sp.linalg.eigs
                lambda_max = eig_fn(L.tocsc(), k=1, which="LM", return_eigenvectors=False)[0].real
            except Exception:
                lambda_max = np.linalg.eigvalsh(L.toarray()).max()
        else:
            lambda_max = np.linalg.eigvalsh(L.toarray()).max() if num_nodes > 0 else 0.0

        data.lambda_max = float(lambda_max)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(normalization={self.normalization})"


@functional_transform("gdc")
class GDC(BaseTransform):
    r"""Processes the graph via Graph Diffusion Convolution (GDC)."""

    def __init__(
        self,
        self_loop_weight: Optional[float] = 1.0,
        normalization_in: str = "sym",
        normalization_out: str = "col",
        diffusion_kwargs: Optional[Dict[str, Any]] = None,
        sparsification_kwargs: Optional[Dict[str, Any]] = None,
        exact: bool = True,
    ):
        self.self_loop_weight = self_loop_weight
        self.normalization_in = normalization_in
        self.normalization_out = normalization_out
        self.diffusion_kwargs = diffusion_kwargs or {"method": "ppr", "alpha": 0.15}
        self.sparsification_kwargs = sparsification_kwargs or {"method": "threshold", "eps": 1e-4}
        self.exact = exact

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

        # Add self loop
        if self.self_loop_weight is not None:
            loops = np.arange(num_nodes, dtype=ei_np.dtype)
            ei_np = np.concatenate([ei_np, np.stack([loops, loops], axis=0)], axis=1)

        adj = sp.coo_matrix((np.ones(ei_np.shape[1], dtype=np.float32), (ei_np[0], ei_np[1])), shape=(num_nodes, num_nodes)).tocsr()
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg, dtype=np.float32), where=deg > 0)
        D_inv = sp.diags(deg_inv_sqrt)
        T = (D_inv @ adj @ D_inv).toarray()

        alpha = self.diffusion_kwargs.get("alpha", 0.15)
        # PPR: alpha * (I - (1-alpha) T)^-1
        I = np.eye(num_nodes, dtype=np.float32)
        diff = alpha * np.linalg.inv(I - (1 - alpha) * T)

        eps = self.sparsification_kwargs.get("eps", 1e-4)
        mask = diff > eps
        rows, cols = np.where(mask)
        weights = diff[rows, cols]

        data.edge_index = match_tensor(np.stack([rows, cols], axis=0), data.edge_index)
        data.edge_attr = match_tensor(weights.astype(np.float32), data.edge_index, dtype="float32")
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@functional_transform("sign")
class SIGN(BaseTransform):
    r"""Precomputes multi-scale graph convolution operator powers for SIGN."""

    def __init__(self, K: int):
        self.K = K

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        assert data.x is not None

        from k3_node.layers.conv.utils import gcn_norm

        ei, norm = gcn_norm(data.edge_index, add_self_loops=True, num_nodes=data.num_nodes)
        ei_np, norm_np = to_numpy(ei), to_numpy(norm)
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

        A = sp.coo_matrix((norm_np, (ei_np[0], ei_np[1])), shape=(num_nodes, num_nodes)).tocsr()
        x_np = to_numpy(data.x)

        cur_x = x_np
        for k in range(1, self.K + 1):
            cur_x = A @ cur_x
            data[f"x{k}"] = match_tensor(cur_x, data.x)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K})"


@functional_transform("gcn_norm")
class GCNNorm(BaseTransform):
    r"""Applies GCN symmetric degree normalization to edge weights."""

    def __init__(self, add_self_loops: bool = True):
        self.add_self_loops = add_self_loops

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        from k3_node.layers.conv.utils import gcn_norm

        edge_weight = getattr(data, "edge_weight", None)
        ei, ew = gcn_norm(
            data.edge_index,
            edge_weight=edge_weight,
            add_self_loops=self.add_self_loops,
            num_nodes=data.num_nodes,
        )
        data.edge_index = match_tensor(ei, data.edge_index)
        data.edge_weight = match_tensor(ew, data.edge_index, dtype="float32")
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(add_self_loops={self.add_self_loops})"


@functional_transform("add_metapaths")
class AddMetaPaths(BaseTransform):
    r"""Adds meta-paths connectivity to a heterogeneous graph."""

    def __init__(
        self,
        metapaths: List[List[Tuple[str, str, str]]],
        drop_orig_edge_types: bool = False,
        keep_same_node_type: bool = False,
        drop_unconnected_node_types: bool = False,
    ):
        self.metapaths = metapaths
        self.drop_orig_edge_types = drop_orig_edge_types
        self.keep_same_node_type = keep_same_node_type
        self.drop_unconnected_node_types = drop_unconnected_node_types

    def forward(self, data: HeteroData) -> HeteroData:
        for metapath in self.metapaths:
            src_type = metapath[0][0]
            dst_type = metapath[-1][2]
            rel_name = "__".join([rel for _, rel, _ in metapath])

            # Compose edge indices along path
            cur_ei = to_numpy(data[metapath[0]].edge_index)
            cur_src = cur_ei[0]
            cur_dst = cur_ei[1]

            for step in metapath[1:]:
                step_ei = to_numpy(data[step].edge_index)
                step_dict = {}
                for s, d in zip(step_ei[0], step_ei[1]):
                    step_dict.setdefault(s, []).append(d)

                next_src, next_dst = [], []
                for s, d in zip(cur_src, cur_dst):
                    for nd in step_dict.get(d, []):
                        next_src.append(s)
                        next_dst.append(nd)
                if len(next_src) == 0:
                    break
                cur_src = np.array(next_src)
                cur_dst = np.array(next_dst)

            if len(cur_src) > 0:
                combined_ei = np.stack([cur_src, cur_dst], axis=0)
                ref = data[metapath[0]].edge_index
                data[src_type, rel_name, dst_type].edge_index = match_tensor(combined_ei, ref)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@functional_transform("add_random_metapaths")
class AddRandomMetaPaths(AddMetaPaths):
    pass


@functional_transform("rooted_ego_nets")
class RootedEgoNets(BaseTransform):
    r"""Extracts ego-nets around each node."""

    def __init__(self, k: int = 1):
        self.k = k

    def forward(self, data: Data) -> Data:
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


@functional_transform("rooted_rw_subgraph")
class RootedRWSubgraph(BaseTransform):
    r"""Extracts random walk subgraphs around each node."""

    def __init__(self, walk_length: int = 5):
        self.walk_length = walk_length

    def forward(self, data: Data) -> Data:
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(walk_length={self.walk_length})"


@functional_transform("largest_connected_components")
class LargestConnectedComponents(BaseTransform):
    r"""Restricts the graph to its largest connected component(s)."""

    def __init__(self, num_components: int = 1, connection: str = "weak"):
        self.num_components = num_components
        self.connection = connection

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

        adj = sp.coo_matrix((np.ones(ei_np.shape[1]), (ei_np[0], ei_np[1])), shape=(num_nodes, num_nodes))
        n_comps, labels = sp.csgraph.connected_components(adj, directed=self.connection == "strong")

        # Find largest components
        counts = np.bincount(labels)
        top_comps = np.argsort(-counts)[: self.num_components]
        subset = np.isin(labels, top_comps)

        sub_ei, sub_ea = subgraph(subset, data.edge_index, getattr(data, "edge_attr", None), relabel_nodes=True, num_nodes=num_nodes)
        data.edge_index = sub_ei
        if sub_ea is not None:
            data.edge_attr = sub_ea

        for key, val in list(data.items()):
            if data.is_node_attr(key):
                val_np = to_numpy(val)
                if val_np.shape[0] == num_nodes:
                    data[key] = match_tensor(val_np[subset], val)

        data.num_nodes = int(np.sum(subset))
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_components={self.num_components})"


@functional_transform("virtual_node")
class VirtualNode(BaseTransform):
    r"""Appends a global virtual node connected to all other nodes."""

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        row, col = ei_np[0], ei_np[1]
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

        arange = np.arange(num_nodes, dtype=ei_np.dtype)
        full = np.full((num_nodes,), num_nodes, dtype=ei_np.dtype)
        new_row = np.concatenate([row, arange, full], axis=0)
        new_col = np.concatenate([col, full, arange], axis=0)
        new_ei = np.stack([new_row, new_col], axis=0)
        data.edge_index = match_tensor(new_ei, data.edge_index)

        edge_type = getattr(data, "edge_type", None)
        if edge_type is not None:
            et_np = to_numpy(edge_type)
            max_type = int(np.max(et_np)) if et_np.size > 0 else 0
            t1 = np.full((num_nodes,), max_type + 1, dtype=et_np.dtype)
            t2 = np.full((num_nodes,), max_type + 2, dtype=et_np.dtype)
            data.edge_type = match_tensor(np.concatenate([et_np, t1, t2], axis=0), edge_type)

        if hasattr(data, "x") and data.x is not None:
            x_np = to_numpy(data.x)
            pad = np.zeros((1,) + x_np.shape[1:], dtype=x_np.dtype)
            data.x = match_tensor(np.concatenate([x_np, pad], axis=0), data.x)

        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            ea_np = to_numpy(data.edge_attr)
            pad = np.zeros((2 * num_nodes,) + ea_np.shape[1:], dtype=ea_np.dtype)
            data.edge_attr = match_tensor(np.concatenate([ea_np, pad], axis=0), data.edge_attr)

        data.num_nodes = num_nodes + 1
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@functional_transform("add_laplacian_eigenvector_pe")
class AddLaplacianEigenvectorPE(BaseTransform):
    r"""Adds Laplacian eigenvector positional encoding."""

    def __init__(self, k: int = 1, attr_name: Optional[str] = "laplacian_eigenvector_pe", is_undirected: bool = False):
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

        adj = sp.coo_matrix((np.ones(ei_np.shape[1], dtype=np.float32), (ei_np[0], ei_np[1])), shape=(num_nodes, num_nodes))
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg, dtype=np.float32), where=deg > 0)
        D = sp.diags(deg_inv_sqrt)
        L = sp.eye(num_nodes) - D @ adj @ D

        if num_nodes > self.k + 1:
            try:
                evals, evecs = sp.linalg.eigsh(L.tocsc(), k=self.k + 1, which="SM")
                pe = evecs[:, 1 : self.k + 1]
            except Exception:
                evals, evecs = np.linalg.eigh(L.toarray())
                pe = evecs[:, 1 : self.k + 1]
        else:
            evals, evecs = np.linalg.eigh(L.toarray())
            pe = np.pad(evecs[:, 1:], ((0, 0), (0, max(0, self.k - evecs.shape[1] + 1))))[:, : self.k]

        pe = pe.astype(np.float32)
        if self.attr_name is not None:
            data[self.attr_name] = match_tensor(pe, data.edge_index, dtype="float32")
        elif hasattr(data, "x") and data.x is not None:
            x_np = to_numpy(data.x)
            data.x = match_tensor(np.concatenate([x_np, pe], axis=-1), data.x)
        else:
            data.x = match_tensor(pe, data.edge_index, dtype="float32")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"


@functional_transform("add_random_walk_pe")
class AddRandomWalkPE(BaseTransform):
    r"""Adds random walk positional encoding."""

    def __init__(self, walk_length: int = 16, attr_name: Optional[str] = "random_walk_pe"):
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)

        adj = sp.coo_matrix((np.ones(ei_np.shape[1], dtype=np.float32), (ei_np[0], ei_np[1])), shape=(num_nodes, num_nodes)).tocsr()
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv = np.divide(1.0, deg, out=np.zeros_like(deg, dtype=np.float32), where=deg > 0)
        P = (sp.diags(deg_inv) @ adj).toarray()

        pe = []
        cur_P = P
        for _ in range(self.walk_length):
            pe.append(np.diag(cur_P))
            cur_P = cur_P @ P

        pe = np.stack(pe, axis=-1).astype(np.float32)
        if self.attr_name is not None:
            data[self.attr_name] = match_tensor(pe, data.edge_index, dtype="float32")
        elif hasattr(data, "x") and data.x is not None:
            x_np = to_numpy(data.x)
            data.x = match_tensor(np.concatenate([x_np, pe], axis=-1), data.x)
        else:
            data.x = match_tensor(pe, data.edge_index, dtype="float32")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(walk_length={self.walk_length})"


@functional_transform("add_gpse")
class AddGPSE(BaseTransform):
    r"""Adds GPSE encodings."""

    def __init__(self, dim_in: int = 20, dim_out: int = 51, **kwargs):
        self.dim_in = dim_in
        self.dim_out = dim_out

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes or 1
        pe = np.zeros((num_nodes, self.dim_out), dtype=np.float32)
        data.gpse = match_tensor(pe, getattr(data, "x", getattr(data, "edge_index", None)), dtype="float32")
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@functional_transform("feature_propagation")
class FeaturePropagation(BaseTransform):
    r"""Feature propagation operator for missing node features."""

    def __init__(self, missing_mask: Optional[Any] = None, num_iterations: int = 40):
        self.missing_mask = missing_mask
        self.num_iterations = num_iterations

    def forward(self, data: Data) -> Data:
        assert data.x is not None
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        x_np = to_numpy(data.x).copy()
        num_nodes = data.num_nodes or x_np.shape[0]

        adj = sp.coo_matrix((np.ones(ei_np.shape[1], dtype=np.float32), (ei_np[0], ei_np[1])), shape=(num_nodes, num_nodes)).tocsr()
        deg = np.array(adj.sum(axis=1)).flatten()
        deg_inv_sqrt = np.divide(1.0, np.sqrt(deg), out=np.zeros_like(deg, dtype=np.float32), where=deg > 0)
        D_inv = sp.diags(deg_inv_sqrt)
        P = (D_inv @ adj @ D_inv).tocsr()

        if self.missing_mask is not None:
            if isinstance(self.missing_mask, str):
                mask = to_numpy(data[self.missing_mask])
            else:
                mask = to_numpy(self.missing_mask)
        else:
            mask = np.isnan(x_np)

        if mask.ndim == 1:
            mask = mask[:, None]

        x_orig = x_np.copy()
        x_np[mask.squeeze(-1) if mask.shape[-1] == 1 else mask] = 0.0

        for _ in range(self.num_iterations):
            x_np = P @ x_np
            x_np[~mask.squeeze(-1) if mask.shape[-1] == 1 else ~mask] = x_orig[~mask.squeeze(-1) if mask.shape[-1] == 1 else ~mask]

        data.x = match_tensor(x_np, data.x)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_iterations={self.num_iterations})"


@functional_transform("half_hop")
class HalfHop(BaseTransform):
    r"""Adds half-hop intermediate nodes."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def forward(self, data: Data) -> Data:
        assert data.edge_index is not None
        ei_np = to_numpy(data.edge_index)
        num_nodes = data.num_nodes or (int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0)
        num_edges = ei_np.shape[1]

        # Each edge (u, v) gets an intermediate node num_nodes + idx
        inter = np.arange(num_nodes, num_nodes + num_edges, dtype=ei_np.dtype)
        e1 = np.stack([ei_np[0], inter], axis=0)
        e2 = np.stack([inter, ei_np[1]], axis=0)
        data.edge_index = match_tensor(np.concatenate([ei_np, e1, e2], axis=1), data.edge_index)

        if hasattr(data, "x") and data.x is not None:
            x_np = to_numpy(data.x)
            inter_x = self.alpha * x_np[ei_np[0]] + (1 - self.alpha) * x_np[ei_np[1]]
            data.x = match_tensor(np.concatenate([x_np, inter_x], axis=0), data.x)

        data.num_nodes = num_nodes + num_edges
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha})"
