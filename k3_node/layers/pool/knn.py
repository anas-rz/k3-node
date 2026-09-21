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
    x = ops.convert_to_tensor(x)
    y = ops.convert_to_tensor(y)

    if len(ops.shape(x)) == 1:
        x = ops.expand_dims(x, axis=-1)
    if len(ops.shape(y)) == 1:
        y = ops.expand_dims(y, axis=-1)

    N = ops.shape(x)[0]
    M = ops.shape(y)[0]
    if cosine:
        x_norm = x / ops.maximum(ops.norm(x, axis=-1, keepdims=True), 1e-12)
        y_norm = y / ops.maximum(ops.norm(y, axis=-1, keepdims=True), 1e-12)
        dist = 1.0 - ops.matmul(y_norm, ops.transpose(x_norm))
    else:
        y_exp = ops.expand_dims(y, axis=1)
        x_exp = ops.expand_dims(x, axis=0)
        dist = ops.sum(ops.power(y_exp - x_exp, 2), axis=-1)

    if batch_x is not None or batch_y is not None:
        batch_x = ops.zeros((N,), dtype="int32") if batch_x is None else ops.cast(batch_x, "int32")
        batch_y = ops.zeros((M,), dtype="int32") if batch_y is None else ops.cast(batch_y, "int32")
        mask = ops.expand_dims(batch_y, axis=1) != ops.expand_dims(batch_x, axis=0)
        dist = ops.where(mask, 1e9, dist)

    _, col_indices = ops.top_k(-dist, k=k, sorted=True)
    row = ops.repeat(ops.arange(0, M, dtype="int64"), k)
    col = ops.reshape(ops.cast(col_indices, "int64"), (-1,))
    return ops.stack([row, col], axis=0)


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
    x = ops.convert_to_tensor(x)
    if len(ops.shape(x)) == 1:
        x = ops.expand_dims(x, axis=-1)

    N = ops.shape(x)[0]
    if cosine:
        x_norm = x / ops.maximum(ops.norm(x, axis=-1, keepdims=True), 1e-12)
        dist = 1.0 - ops.matmul(x_norm, ops.transpose(x_norm))
    else:
        x_exp_0 = ops.expand_dims(x, axis=0)
        x_exp_1 = ops.expand_dims(x, axis=1)
        dist = ops.sum(ops.power(x_exp_1 - x_exp_0, 2), axis=-1)

    if not loop:
        diag_mask = ops.eye(N, dtype=dist.dtype) * 1e9
        dist = dist + diag_mask

    if batch is not None:
        batch = ops.cast(batch, "int32")
        batch_mask = ops.expand_dims(batch, axis=1) != ops.expand_dims(batch, axis=0)
        dist = ops.where(batch_mask, 1e9, dist)

    _, col_indices = ops.top_k(-dist, k=k, sorted=True)
    row = ops.repeat(ops.arange(0, N, dtype="int64"), k)
    col = ops.reshape(ops.cast(col_indices, "int64"), (-1,))

    if flow == "source_to_target":
        return ops.stack([col, row], axis=0)
    else:
        return ops.stack([row, col], axis=0)

