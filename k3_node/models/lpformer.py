"""LPFormer, ported from PyG (``torch_geometric.nn.models.LPFormer``).

The choice of the nodes each candidate link attends to (common neighbors, one-hop neighbors and
nodes with a high personalized PageRank) depends on the graph only and is computed on the host
with SciPy sparse matrices; the learnable parts run with Keras ops. The model runs eagerly.
"""
import math
from typing import List, Optional

import keras
import numpy as np
from keras import ops

from k3_node.models.basic_gnn import GCN
from k3_node.ops.segment import segment_sum


def get_ppr(edge_index, num_nodes: int, alpha: float = 0.15, eps: float = 5e-5):
    r"""Approximate personalized PageRank of every node (the Andersen push algorithm used by
    PyG's ``get_ppr``). Returns a ``scipy.sparse.csr_matrix`` whose row ``i`` holds the PPR scores
    of the nodes reachable from ``i``.

    Example:
        ```python
        import numpy as np
        from k3_node.models.lpformer import get_ppr

        edge_index = np.array([[0, 1, 1, 2], [1, 0, 2, 1]])  # a path 0 - 1 - 2
        ppr = get_ppr(edge_index, num_nodes=3)
        print(ppr.shape, round(float(ppr[0, 0]), 2))  # (3, 3) 0.19
        ```
    """
    import scipy.sparse as sp

    from k3_node.ops.host import to_numpy

    edge_index = np.asarray(to_numpy(edge_index)).astype(np.int64)
    order = np.lexsort((edge_index[1], edge_index[0]))  # CSR with sorted columns, as PyG's EdgeIndex
    col = edge_index[1][order]
    rowptr = np.concatenate([[0], np.cumsum(np.bincount(edge_index[0], minlength=num_nodes))])
    cols_list, vals_list = _ppr_push(rowptr, col, alpha, eps)
    rows = np.repeat(np.arange(num_nodes), [len(c) for c in cols_list])
    cols = np.concatenate(cols_list) if cols_list else np.zeros(0, np.int64)
    vals = np.concatenate(vals_list) if vals_list else np.zeros(0)
    return sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(num_nodes, num_nodes))


def _ppr_push_python(rowptr, col, alpha, eps):
    """PyG's Andersen push (``torch_geometric.utils.ppr._get_ppr``), one source node at a time."""
    alpha_eps = alpha * eps
    cols_list, vals_list = [], []
    for inode in range(len(rowptr) - 1):
        p, r, q, in_q = {inode: 0.0}, {inode: alpha}, [inode], {inode}
        while q:
            unode = q.pop()
            in_q.discard(unode)
            res = r.get(unode, 0.0)
            p[unode] = p.get(unode, 0.0) + res
            r[unode] = 0.0
            start, end = rowptr[unode], rowptr[unode + 1]
            ucount = end - start
            for vnode in col[start:end]:
                vnode = int(vnode)
                r[vnode] = r.get(vnode, 0.0) + (1 - alpha) * res / ucount
                if r[vnode] >= alpha_eps * (rowptr[vnode + 1] - rowptr[vnode]) and vnode not in in_q:
                    q.append(vnode)
                    in_q.add(vnode)
        cols_list.append(np.fromiter(p.keys(), dtype=np.int64, count=len(p)))
        vals_list.append(np.fromiter(p.values(), dtype=np.float64, count=len(p)))
    return cols_list, vals_list


_PPR_NUMBA = None


def _ppr_push(rowptr, col, alpha, eps):
    """Runs the push algorithm compiled with numba when it is installed (as PyG does)."""
    global _PPR_NUMBA
    try:
        import numba
    except ImportError:
        return _ppr_push_python(rowptr, col, alpha, eps)
    if _PPR_NUMBA is None:
        def push(rowptr, col, alpha, eps):
            num_nodes = len(rowptr) - 1
            alpha_eps = alpha * eps
            js = [[0]] * num_nodes
            vals = [[0.0]] * num_nodes
            for inode_uint in numba.prange(num_nodes):
                inode = numba.int64(inode_uint)
                p = {inode: 0.0}
                r = {}
                r[inode] = alpha
                q = [inode]
                while len(q) > 0:
                    unode = q.pop()
                    res = r[unode] if unode in r else 0
                    if unode in p:
                        p[unode] += res
                    else:
                        p[unode] = res
                    r[unode] = 0
                    start, end = rowptr[unode], rowptr[unode + 1]
                    ucount = end - start
                    for vnode in col[start:end]:
                        _val = (1 - alpha) * res / ucount
                        if vnode in r:
                            r[vnode] += _val
                        else:
                            r[vnode] = _val
                        res_vnode = r[vnode] if vnode in r else 0
                        vcount = rowptr[vnode + 1] - rowptr[vnode]
                        if res_vnode >= alpha_eps * vcount:
                            if vnode not in q:
                                q.append(vnode)
                js[inode_uint] = list(p.keys())
                vals[inode_uint] = list(p.values())
            return js, vals

        _PPR_NUMBA = numba.jit(nopython=True, parallel=True)(push)
    js, vals = _PPR_NUMBA(rowptr, col, alpha, eps)
    return [np.asarray(j, dtype=np.int64) for j in js], [np.asarray(v, dtype=np.float64) for v in vals]


def compute_ppr_matrix(edge_index, num_nodes: int, alpha: float = 0.15, eps: float = 5e-5):
    r"""Alias of :func:`get_ppr`."""
    return get_ppr(edge_index, num_nodes, alpha=alpha, eps=eps)


def _lookup(matrix, rows, cols):
    """``matrix[rows[k], cols[k]]`` for every ``k``, as a 1-D array (also when empty)."""
    if len(rows) == 0:
        return np.zeros(0, dtype=np.float32)
    values = matrix[rows, cols]
    values = values.toarray() if hasattr(values, "toarray") else values
    return np.asarray(values, dtype=np.float32).ravel()


class MLP(keras.layers.Layer):
    r"""The small MLP of LPFormer: linear layers with (layer) normalization, ReLU and dropout
    between them; the last dimension is squeezed if it has size 1."""

    def __init__(self, in_channels: int, hid_channels: int, out_channels: int, num_layers: int = 2,
                 drop: float = 0.0, norm: Optional[str] = "layer", **kwargs):
        super().__init__(**kwargs)
        self.linears = ([keras.layers.Dense(out_channels)] if num_layers == 1 else
                        [keras.layers.Dense(hid_channels) for _ in range(num_layers - 1)] + [keras.layers.Dense(out_channels)])
        if norm == "batch":
            self.norm = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        elif norm == "layer":
            self.norm = keras.layers.LayerNormalization(epsilon=1e-5)
        else:
            self.norm = None
        self.dropout = keras.layers.Dropout(drop)

    def call(self, x, training=False):
        for lin in self.linears[:-1]:
            x = lin(x)
            x = self.norm(x, training=training) if self.norm is not None else x
            x = self.dropout(ops.relu(x), training=training)
        x = self.linears[-1](x)
        return ops.squeeze(x, axis=-1) if x.shape[-1] == 1 else x


class LPAttLayer(keras.layers.Layer):
    r"""Attention of every candidate link over its selected nodes (PyG's ``LPAttLayer``).

    Used inside :class:`LPFormer`. ``edge_index[0]`` is the link and ``edge_index[1]`` a node it
    attends to; ``edge_feats`` holds the two end-node features of every link side by side and
    ``ppr_rpes`` a relative positional encoding for every (link, node) pair.

    Example:
        ```python
        import numpy as np
        from k3_node.models import LPAttLayer

        layer = LPAttLayer(in_channels=8, out_channels=8, node_dim=None, num_heads=2, dropout=0.0)
        edge_feats = np.random.rand(4, 16).astype("float32")  # 4 links: [x_src | x_dst]
        node_feats = np.random.rand(10, 8).astype("float32")  # 10 nodes
        edge_index = np.stack([np.repeat(np.arange(4), 3), np.random.randint(0, 10, size=12)])
        ppr_rpes = np.random.rand(12, 8).astype("float32")  # one encoding per (link, node) pair
        out = layer(edge_index, edge_feats, node_feats, ppr_rpes)
        print(tuple(out.shape))  # (4, 16)
        ```
    """

    def __init__(self, in_channels: int, out_channels: int, node_dim: Optional[int], num_heads: int,
                 dropout: float, concat: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.in_channels, self.out_channels, self.heads, self.concat = in_channels, out_channels, num_heads, concat
        self.negative_slope = 0.2
        self.lin_l = keras.layers.Dense(num_heads * out_channels, kernel_initializer="glorot_uniform")
        self.lin_r = keras.layers.Dense(num_heads * out_channels, kernel_initializer="glorot_uniform")
        self.att = self.add_weight(shape=(1, num_heads, out_channels), initializer="glorot_uniform", name="att")
        self.bias = self.add_weight(shape=(num_heads * out_channels if concat else out_channels,),
                                    initializer="zeros", name="bias")
        self.post_att_norm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.dropout = keras.layers.Dropout(dropout)

    def call(self, edge_index, edge_feats, node_feats, ppr_rpes, training=False):
        H, C = self.heads, self.out_channels
        pair, node = edge_index[0], edge_index[1]  # "target_to_source": link i attends to node j
        num_links = edge_feats.shape[0]
        x_i = ops.take(edge_feats, pair, axis=0)
        x_j = ops.concatenate([ops.take(node_feats, node, axis=0), ppr_rpes], axis=-1)
        x_j = ops.reshape(self.lin_r(x_j), (-1, H, C))
        e1, e2 = ops.split(x_i, 2, axis=-1)
        x = ops.leaky_relu(x_j * (ops.reshape(self.lin_l(e1), (-1, H, C)) + ops.reshape(self.lin_l(e2), (-1, H, C))),
                           negative_slope=self.negative_slope)
        alpha = ops.sum(x * self.att, axis=-1)  # [K, H]
        from k3_node.layers.conv.utils import softmax

        alpha = softmax(alpha, pair, num_nodes=num_links)
        out = segment_sum(x_j * ops.expand_dims(alpha, -1), pair, num_segments=num_links)  # [B, H, C]
        out = ops.reshape(out, (-1, H * C)) if self.concat else ops.mean(out, axis=1)
        out = self.post_att_norm(out + self.bias)
        return self.dropout(out, training=training)


class LPFormer(keras.Model):
    r"""The LPFormer model from the `"LPFormer: An Adaptive Graph Transformer for Link Prediction"
    <https://arxiv.org/abs/2310.11009>`_ paper, as in PyG.

    For every candidate link it attends over the common neighbors, the one-hop neighbors and the
    other nodes with a high personalized PageRank (PPR) from both endpoints (``ppr_thresholds``
    for the three kinds), using their PPR scores as relative positional encodings, and combines
    this with counts of each kind of node and with the GCN embeddings of the two endpoints.

    Args:
        in_channels (int): Input feature dimension.
        hidden_channels (int): Hidden dimension.
        num_gnn_layers (int, optional): Number of GCN layers. (default: ``2``)
        gnn_dropout (float, optional): GCN dropout. (default: ``0.1``)
        num_transformer_layers (int, optional): Number of attention layers. (default: ``1``)
        num_heads (int, optional): Number of attention heads. (default: ``1``)
        transformer_dropout (float, optional): Attention dropout; during training this share of
            the selected nodes is also dropped. (default: ``0.1``)
        ppr_thresholds (list, optional): Minimum PPR of common neighbors, one-hop neighbors and
            other nodes. (default: ``[0, 1e-4, 1e-2]``)

    Call arguments: ``batch`` (the ``[2, num_links]`` candidate links), ``x`` (node features),
    ``edge_index`` (the graph) and the keyword ``ppr_matrix`` (from :meth:`calc_sparse_ppr`). Returns one logit
    per link. The node selection runs on the host, so train with ``run_eagerly=True`` or
    :func:`~k3_node.training.gradient_step`.

    Example:
        ```python
        import numpy as np
        from k3_node.models import LPFormer

        x = np.random.rand(10, 16).astype("float32")  # 10 nodes with 16 features each
        edge_index = np.random.randint(0, 10, size=(2, 30))  # 30 random edges
        model = LPFormer(in_channels=16, hidden_channels=16)
        ppr = model.calc_sparse_ppr(edge_index, num_nodes=10)
        target_links = np.array([[0, 1], [2, 3]])  # the (source, target) pairs to score
        print(tuple(model(target_links, x, edge_index, ppr_matrix=ppr).shape))  # (2,): one logit per link
        ```
    """

    def __init__(self, in_channels: int, hidden_channels: int, num_gnn_layers: int = 2, gnn_dropout: float = 0.1,
                 num_transformer_layers: int = 1, num_heads: int = 1, transformer_dropout: float = 0.1,
                 ppr_thresholds: Optional[List[float]] = None, **kwargs):
        super().__init__(**kwargs)
        ppr_thresholds = [0, 1e-4, 1e-2] if ppr_thresholds is None else ppr_thresholds
        if len(ppr_thresholds) != 3:
            raise ValueError("Argument 'ppr_thresholds' must only be length 3!")
        self.thresh_cn, self.thresh_1hop, self.thresh_non1hop = ppr_thresholds
        self.in_dim, self.hid_dim = in_channels, hidden_channels
        self.trans_drop = transformer_dropout
        self.gnn = GCN(in_channels, hidden_channels, num_gnn_layers, dropout=gnn_dropout, norm="layer_norm")
        self.gnn_norm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.input_dropout = keras.layers.Dropout(gnn_dropout)
        self.att_layers = []
        for il in range(num_transformer_layers):
            if il == 0:
                node_dim = None
                self.out_dim = hidden_channels * 2 if num_transformer_layers > 1 else hidden_channels
            else:
                self.out_dim = node_dim = hidden_channels
            self.att_layers.append(LPAttLayer(hidden_channels, self.out_dim, node_dim, num_heads, transformer_dropout))
        self.elementwise_lin = MLP(hidden_channels, hidden_channels, hidden_channels)
        self.ppr_encoder_cn = MLP(2, hidden_channels, hidden_channels)
        self.ppr_encoder_onehop = MLP(2, hidden_channels, hidden_channels)
        self.ppr_encoder_non1hop = MLP(2, hidden_channels, hidden_channels)
        if self.thresh_non1hop == 1 and self.thresh_1hop == 1:
            self.mask = "cn"
        elif self.thresh_non1hop == 1 and self.thresh_1hop < 1:
            self.mask = "1-hop"
        else:
            self.mask = "all"
        pairwise_dim = hidden_channels * num_heads + 4
        self.pairwise_lin = MLP(pairwise_dim, pairwise_dim, hidden_channels)
        self.score_func = MLP(hidden_channels * 2, hidden_channels * 2, 1, norm=None)

    @staticmethod
    def calc_sparse_ppr(edge_index, num_nodes: int, alpha: float = 0.15, eps: float = 5e-5):
        r"""The personalized PageRank matrix LPFormer needs (see :func:`get_ppr`)."""
        return get_ppr(edge_index, num_nodes, alpha=alpha, eps=eps)

    # ---- node selection on the host ----------------------------------------------------------
    def _drop(self, rng, n):
        keep = math.ceil(n * (1 - self.trans_drop))
        return rng.permutation(n)[:keep]

    def compute_node_mask(self, u, v, adj, ppr, training):
        r"""For the links ``(u, v)``: ``(pair index, node, PPR from u, PPR from v)`` of their common
        neighbors, one-hop neighbors and other high-PPR nodes."""
        pair_adj = (adj[u] * adj[v]) if self.mask == "cn" else (adj[u] + adj[v])
        pair_adj = pair_adj.tocoo()
        order = np.lexsort((pair_adj.col, pair_adj.row))
        rows, cols, node_type = pair_adj.row[order], pair_adj.col[order], pair_adj.data[order]
        src_ppr = _lookup(ppr, u[rows], cols)
        tgt_ppr = _lookup(ppr, v[rows], cols)
        cn_cond = (src_ppr >= self.thresh_cn) & (tgt_ppr >= self.thresh_cn)
        onehop_cond = (src_ppr >= self.thresh_1hop) & (tgt_ppr >= self.thresh_1hop)
        keep = np.where(node_type == 1, onehop_cond, cn_cond) if self.mask != "cn" else np.where(node_type == 0, onehop_cond, cn_cond)
        rows, cols, node_type, src_ppr, tgt_ppr = rows[keep], cols[keep], node_type[keep], src_ppr[keep], tgt_ppr[keep]

        non1hop = None
        if self.mask == "all":
            non1hop = self._non_1hop(u, v, adj, ppr, training)
        rng = np.random
        if training and self.trans_drop > 0:
            idx = self._drop(rng, len(rows))
            rows, cols, node_type, src_ppr, tgt_ppr = rows[idx], cols[idx], node_type[idx], src_ppr[idx], tgt_ppr[idx]
            if non1hop is not None:
                idx = self._drop(rng, len(non1hop[0]))
                non1hop = tuple(a[idx] for a in non1hop)
        if self.mask == "cn":
            return (rows, cols, src_ppr, tgt_ppr), None, None
        cn = node_type == 2
        one = node_type == 1
        return ((rows[cn], cols[cn], src_ppr[cn], tgt_ppr[cn]), (rows[one], cols[one], src_ppr[one], tgt_ppr[one]),
                non1hop)

    def _non_1hop(self, u, v, adj, ppr, training):
        import scipy.sparse as sp

        adj2 = adj
        if training:  # the links being predicted are known edges during training
            n = adj.shape[0]
            links = sp.csr_matrix((np.ones(2 * len(u)), (np.concatenate([u, v]), np.concatenate([v, u]))), shape=(n, n))
            adj2 = ((adj + links) > 0).astype(np.float32).tocsr()
        neighbors = ((adj2[u] + adj2[v]) > 0).astype(np.float32)
        src, tgt = ppr[u], ppr[v]
        both = (src >= self.thresh_non1hop).astype(np.float32).multiply((tgt >= self.thresh_non1hop).astype(np.float32))
        both = sp.csr_matrix(both - both.multiply(neighbors))  # high PPR from both ends, not a neighbor
        both.eliminate_zeros()
        both = both.tocoo()
        order = np.lexsort((both.col, both.row))
        rows, cols = both.row[order], both.col[order]
        return rows, cols, _lookup(ppr, u[rows], cols), _lookup(ppr, v[rows], cols)

    # ---- forward ------------------------------------------------------------------------------------
    def _pos_encoding(self, encoder, s, t, training):
        a = ops.convert_to_tensor(np.stack([s, t], axis=1).astype(np.float32))
        b = ops.convert_to_tensor(np.stack([t, s], axis=1).astype(np.float32))
        return encoder(a, training=training) + encoder(b, training=training)

    def call(self, batch, x, edge_index, ppr_matrix=None, training=False):
        import scipy.sparse as sp

        from k3_node.ops.host import to_numpy

        batch_np = np.asarray(to_numpy(batch)).astype(np.int64)
        edge_np = np.asarray(to_numpy(edge_index)).astype(np.int64)
        num_nodes = x.shape[0]
        if ppr_matrix is None:
            ppr_matrix = self.calc_sparse_ppr(edge_np, num_nodes)
        ppr = sp.csr_matrix(ppr_matrix)
        adj = sp.csr_matrix((np.ones(edge_np.shape[1], np.float32), (edge_np[0], edge_np[1])), shape=(num_nodes, num_nodes))
        adj.data[:] = 1.0  # {0, 1} even with duplicate edges

        X_node = self.gnn_norm(self.gnn(self.input_dropout(x, training=training), edge_index, training=training))
        u, v = batch_np[0], batch_np[1]
        x_i, x_j = ops.take(X_node, u, axis=0), ops.take(X_node, v, axis=0)
        elementwise = self.elementwise_lin(x_i * x_j, training=training)

        cn, onehop, non1hop = self.compute_node_mask(u, v, adj, ppr, training)
        groups = [(cn, self.ppr_encoder_cn), (onehop, self.ppr_encoder_onehop), (non1hop, self.ppr_encoder_non1hop)]
        groups = [(g, enc) for g, enc in groups if g is not None]
        rows = np.concatenate([g[0] for g, _ in groups])
        cols = np.concatenate([g[1] for g, _ in groups])
        pes = ops.concatenate([self._pos_encoding(enc, g[2], g[3], training) for g, enc in groups], axis=0)
        all_mask = ops.convert_to_tensor(np.stack([rows, cols]).astype(np.int32))

        pairwise = ops.concatenate([x_i, x_j], axis=-1)
        for layer in self.att_layers:
            pairwise = layer(all_mask, pairwise, X_node, pes, training=training)

        B = len(u)
        counts = [np.bincount(cn[0], minlength=B)]  # common neighbors (all pass thresh_cn)
        if onehop is not None:
            num_1hop = np.bincount(onehop[0][(onehop[2] >= self.thresh_1hop) & (onehop[3] >= self.thresh_1hop)], minlength=B)
            num_ppr_ones = np.bincount(onehop[0], minlength=B)
            counts += [num_1hop]
        else:
            num_ppr_ones = np.zeros(B)
            counts += [np.zeros(B)]
        counts += [np.bincount(non1hop[0], minlength=B) if non1hop is not None else np.zeros(B), counts[0] + num_ppr_ones]
        counts = ops.convert_to_tensor(np.stack(counts, axis=1).astype(np.float32))

        pairwise = self.pairwise_lin(ops.concatenate([pairwise, counts], axis=-1), training=training)
        return self.score_func(ops.concatenate([elementwise, pairwise], axis=-1), training=training)
