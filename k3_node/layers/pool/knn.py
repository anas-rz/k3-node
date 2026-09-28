from typing import NamedTuple, Optional
from keras import ops
import numpy as np
from k3_node.ops.creation import repeat


class KNNOutput(NamedTuple):
    score: any
    index: any


class KNNIndex:
    r"""A base class to perform k-nearest neighbor search."""
    def __init__(
        self,
        index_factory: Optional[str] = None,
        emb: Optional[any] = None,
        reserve: Optional[int] = None,
    ):
        self.index_factory = index_factory
        self.reserve = reserve
        self.emb = None
        if emb is not None:
            self.add(emb)

    @property
    def numel(self) -> int:
        if self.emb is None:
            return 0
        return ops.shape(self.emb)[0]

    def add(self, emb):
        if self.emb is None:
            self.emb = emb
        else:
            self.emb = ops.concatenate([self.emb, emb], axis=0)

    def get_emb(self):
        return self.emb

    def search(self, query, k: int) -> KNNOutput:
        raise NotImplementedError


class L2KNNIndex(KNNIndex):
    r"""k-NN search using squared Euclidean (L2) distance."""
    def __init__(self, emb: Optional[any] = None, **kwargs):
        super().__init__(index_factory="IndexFlatL2", emb=emb, **kwargs)

    def search(self, query, k: int) -> KNNOutput:
        # query: [M, F], emb: [N, F]
        q_exp = ops.expand_dims(query, axis=1)  # [M, 1, F]
        e_exp = ops.expand_dims(self.emb, axis=0)  # [1, N, F]
        dist = ops.sum(ops.power(q_exp - e_exp, 2), axis=-1)  # [M, N]

        top_k_neg_score, top_k_idx = ops.top_k(-dist, k=k, sorted=True)
        return KNNOutput(score=-top_k_neg_score, index=top_k_idx)


class MIPSKNNIndex(KNNIndex):
    r"""k-NN search using Maximum Inner Product Search (MIPS)."""
    def __init__(self, emb: Optional[any] = None, **kwargs):
        super().__init__(index_factory="IndexFlatIP", emb=emb, **kwargs)

    def search(self, query, k: int) -> KNNOutput:
        # query: [M, F], emb: [N, F]
        score_mat = ops.matmul(query, ops.transpose(self.emb))  # [M, N]
        top_k_score, top_k_idx = ops.top_k(score_mat, k=k, sorted=True)
        return KNNOutput(score=top_k_score, index=top_k_idx)


class ApproxL2KNNIndex(L2KNNIndex):
    pass


class ApproxMIPSKNNIndex(MIPSKNNIndex):
    pass


def knn(
    x,
    y,
    k: int,
    batch_x: Optional[any] = None,
    batch_y: Optional[any] = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
):
    r"""Finds for each element in `y` the `k` nearest points in `x`.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import knn

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        pos = np.random.rand(10, 3).astype("float32")  # 3D positions
        query = np.random.rand(4, 3).astype("float32")  # 4 query points

        assign = knn(pos, query, k=3)  # 3 nearest points in `pos` for every query point
        print(tuple(assign.shape))  # (2, 12): [2, num_query * k]: (query index, point index)
        ```
    """
    x = ops.convert_to_tensor(x)
    y = ops.convert_to_tensor(y)
    if len(ops.shape(x)) == 1:
        x = ops.expand_dims(x, axis=-1)
    if len(ops.shape(y)) == 1:
        y = ops.expand_dims(y, axis=-1)

    col = _knn_indices(x, y, k, batch_x, batch_y, cosine=cosine)  # [M, k] indices into x
    M = ops.shape(y)[0]
    row = repeat(ops.arange(0, M, dtype="int64"), k)
    return ops.stack([row, ops.reshape(ops.cast(col, "int64"), (-1,))], axis=0)


def _pairwise(y, x, cosine):
    """Squared distances (or cosine distances) between the rows of ``y`` and ``x``, without
    materializing ``[M, N, features]`` differences."""
    if cosine:
        y = y / ops.maximum(ops.norm(y, axis=-1, keepdims=True), 1e-12)
        x = x / ops.maximum(ops.norm(x, axis=-1, keepdims=True), 1e-12)
        return 1.0 - ops.matmul(y, ops.swapaxes(x, -1, -2))
    y_sq = ops.sum(y * y, axis=-1, keepdims=True)
    x_sq = ops.expand_dims(ops.sum(x * x, axis=-1), -2)
    return ops.maximum(y_sq + x_sq - 2.0 * ops.matmul(y, ops.swapaxes(x, -1, -2)), 0.0)


def _knn_indices(x, y, k, batch_x=None, batch_y=None, cosine=False, exclude_self=False):
    """Indices ``[M, k]`` into ``x`` of the ``k`` nearest points of every row of ``y`` within the
    same example. Eagerly, distances are computed per example (``[examples, n, n]``); when compiled,
    in chunks of query rows masked by example."""
    from k3_node.layers.aggr.base import to_dense_batch
    from k3_node.layers.conv.utils import is_tracing

    N, M = ops.shape(x)[0], ops.shape(y)[0]
    if isinstance(M, int) and isinstance(N, int) and (M == 0 or N == 0):  # sizes are tensors while tracing
        return ops.zeros((M, k), dtype="int32")
    ids = [b for b in (batch_x, batch_y) if b is not None]
    if not any(is_tracing(t) for t in [x, y] + ids) and isinstance(N, int) and isinstance(M, int):
        try:
            bx = np.zeros(N, np.int64) if batch_x is None else np.asarray(ops.convert_to_numpy(batch_x)).astype(np.int64)
            by = np.zeros(M, np.int64) if batch_y is None else np.asarray(ops.convert_to_numpy(batch_y)).astype(np.int64)
            num_examples = int(max(bx.max(initial=-1), by.max(initial=-1))) + 1
            x_dense, x_mask = to_dense_batch(x, bx, dim_size=num_examples)
            y_dense, _ = to_dense_batch(y, by, dim_size=num_examples)
            dist = _pairwise(y_dense, x_dense, cosine)  # [examples, max_y, max_x]
            dist = ops.where(ops.expand_dims(x_mask, 1), dist, 1e9)
            x_start = np.concatenate([[0], np.cumsum(np.bincount(bx, minlength=num_examples))])[:-1]
            y_start = np.concatenate([[0], np.cumsum(np.bincount(by, minlength=num_examples))])[:-1]
            y_local = np.arange(M) - y_start[by]
            if exclude_self:  # x and y are the same points
                dist = ops.where(ops.expand_dims(ops.eye(dist.shape[1], dist.shape[2]), 0) > 0, 1e9, dist)
            kk = min(k, int(dist.shape[2]))
            _, idx = ops.top_k(-dist, k=kk, sorted=True)  # local indices [examples, max_y, kk]
            flat = ops.reshape(idx, (-1, kk))
            local = ops.take(flat, by * int(dist.shape[1]) + y_local, axis=0)  # [M, kk]
            out = local + ops.convert_to_tensor(x_start[by][:, None].astype(np.int64), dtype=local.dtype)
            if kk < k:  # fewer candidates than k: repeat the last one
                out = ops.concatenate([out] + [out[:, -1:]] * (k - kk), axis=1)
            return out
        except (TypeError, ValueError, NotImplementedError, RuntimeError):
            pass

    if y is x and batch_x is not None and batch_y is batch_x:  # compiled knn_graph: per example
        return _knn_graph_indices_traced(x, k, ops.cast(batch_x, "int32"), cosine, exclude_self)

    # Compiled: masked distances for chunks of query rows ([chunk, N] memory at a time)
    bx = ops.zeros((N,), "int32") if batch_x is None else ops.cast(batch_x, "int32")
    by = ops.zeros((M,), "int32") if batch_y is None else ops.cast(batch_y, "int32")
    chunk = M if not isinstance(M, int) or not isinstance(N, int) else max(1, min(M, (1 << 24) // max(N, 1)))
    outs = []
    for start in range(0, M, chunk) if isinstance(M, int) else [0]:
        stop = start + chunk if isinstance(M, int) else None
        dist = _pairwise(y[start:stop], x, cosine)
        mask = ops.expand_dims(by[start:stop], 1) != ops.expand_dims(bx, 0)
        if exclude_self:
            rows = ops.arange(ops.shape(dist)[0], dtype="int32") + start
            mask = ops.logical_or(mask, ops.expand_dims(rows, 1) == ops.expand_dims(ops.arange(N, dtype="int32"), 0))
        _, idx = ops.top_k(-ops.where(mask, 1e9, dist), k=k, sorted=True)
        outs.append(idx)
    return outs[0] if len(outs) == 1 else ops.concatenate(outs, axis=0)


def _knn_graph_indices_traced(x, k, batch, cosine, exclude_self):
    """``_knn_indices(x, x, ...)`` for sizes only known at run time: distances within each example
    (``[examples, max_points, max_points]``), like the eager path. Needs ``k`` <= points per example."""
    from k3_node.layers.aggr.base import to_dense_batch
    from k3_node.ops.segment import segment_sum

    num_points = ops.shape(x)[0]
    x_dense, mask = to_dense_batch(x, batch)
    n = ops.shape(x_dense)[1]
    dist = ops.where(ops.expand_dims(mask, 1), _pairwise(x_dense, x_dense, cosine), 1e9)
    if exclude_self:
        same = ops.expand_dims(ops.arange(n), 1) == ops.expand_dims(ops.arange(n), 0)
        dist = ops.where(ops.expand_dims(same, 0), 1e9, dist)
    _, idx = ops.top_k(-dist, k=k, sorted=True)  # local indices [examples, max_points, k]
    counts = segment_sum(ops.ones_like(batch), batch, num_segments=num_points)
    starts = ops.take(ops.cumsum(counts) - counts, batch, axis=0)  # first point of each point's example
    local = ops.arange(num_points, dtype="int32") - starts
    rows = ops.take(ops.reshape(idx, (-1, k)), batch * ops.cast(n, "int32") + local, axis=0)
    return rows + ops.cast(ops.expand_dims(starts, 1), rows.dtype)


def knn_graph(
    x,
    k: int,
    batch: Optional[any] = None,
    loop: bool = False,
    flow: str = "source_to_target",
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
):
    r"""Computes graph edges to the nearest `k` points.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import knn_graph

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        pos = np.random.rand(10, 3).astype("float32")  # 3D positions

        edge_index = knn_graph(pos, k=3)  # connect every point to its 3 nearest neighbors
        print(tuple(edge_index.shape))  # (2, 30)
        ```
    """
    assert flow in ["source_to_target", "target_to_source"]
    x = ops.convert_to_tensor(x)
    if len(ops.shape(x)) == 1:
        x = ops.expand_dims(x, axis=-1)

    N = ops.shape(x)[0]
    col_indices = _knn_indices(x, x, k, batch, batch, cosine=cosine, exclude_self=not loop)
    row = repeat(ops.arange(0, N, dtype="int64"), k)
    col = ops.reshape(ops.cast(col_indices, "int64"), (-1,))

    if flow == "source_to_target":
        return ops.stack([col, row], axis=0)
    else:
        return ops.stack([row, col], axis=0)

