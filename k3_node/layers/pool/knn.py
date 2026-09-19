from typing import NamedTuple, Optional
from keras import ops
import numpy as np


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
    r"""Finds for each element in `y` the `k` nearest points in `x`."""
    if len(ops.shape(x)) == 1:
        x = ops.expand_dims(x, axis=-1)
    if len(ops.shape(y)) == 1:
        y = ops.expand_dims(y, axis=-1)

    N = ops.shape(x)[0]
    M = ops.shape(y)[0]

    x_np = ops.convert_to_numpy(x)
    y_np = ops.convert_to_numpy(y)

    if cosine:
        x_norm = x_np / np.maximum(np.linalg.norm(x_np, axis=-1, keepdims=True), 1e-12)
        y_norm = y_np / np.maximum(np.linalg.norm(y_np, axis=-1, keepdims=True), 1e-12)
        dist = 1.0 - np.dot(y_norm, x_norm.T)
    else:
        # Pairwise Euclidean squared distance
        y_sq = np.sum(y_np**2, axis=-1, keepdims=True)
        x_sq = np.sum(x_np**2, axis=-1, keepdims=True)
        dist = np.maximum(y_sq + x_sq.T - 2.0 * np.dot(y_np, x_np.T), 0.0)

    if batch_x is not None or batch_y is not None:
        if batch_x is None:
            batch_x_np = np.zeros(N, dtype=np.int64)
        else:
            batch_x_np = ops.convert_to_numpy(batch_x).astype(np.int64)

        if batch_y is None:
            batch_y_np = np.zeros(M, dtype=np.int64)
        else:
            batch_y_np = ops.convert_to_numpy(batch_y).astype(np.int64)

        mask = batch_y_np[:, None] != batch_x_np[None, :]
        dist[mask] = float("inf")

    # For each row in y, find k smallest
    k_actual = min(k, N)
    col_indices = np.argsort(dist, axis=-1)[:, :k_actual]

    row = np.repeat(np.arange(M), k_actual)
    col = col_indices.reshape(-1)

    # Filter out infinite distances
    valid = ~np.isinf(dist[row, col])
    row = row[valid]
    col = col[valid]

    return ops.convert_to_tensor(np.stack([row, col], axis=0), dtype="int64")


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
    r"""Computes graph edges to the nearest `k` points."""
    assert flow in ["source_to_target", "target_to_source"]
    edge_index = knn(
        x,
        x,
        k if loop else k + 1,
        batch_x=batch,
        batch_y=batch,
        cosine=cosine,
    )
    edge_index_np = ops.convert_to_numpy(edge_index)
    if not loop:
        mask = edge_index_np[0] != edge_index_np[1]
        edge_index_np = edge_index_np[:, mask]

    if flow == "source_to_target":
        edge_index_np = np.flip(edge_index_np, axis=0)

    return ops.convert_to_tensor(edge_index_np, dtype=edge_index.dtype)

