from typing import Optional
from keras import ops
import numpy as np

from .knn import knn


def fps(
    x,
    batch: Optional[any] = None,
    ratio: float = 0.5,
    random_start: bool = True,
    batch_size: Optional[int] = None,
):
    r"""Farthest Point Sampling algorithm."""
    x_np = ops.convert_to_numpy(x)
    num_nodes = x_np.shape[0]

    if batch is None:
        batch_np = np.zeros(num_nodes, dtype=np.int64)
    else:
        batch_np = ops.convert_to_numpy(batch).astype(np.int64)

    unique_batches = np.unique(batch_np)
    selected_indices = []

    for b in unique_batches:
        idx_b = np.where(batch_np == b)[0]
        n_b = len(idx_b)
        if n_b == 0:
            continue

        if ratio >= 1:
            num_samples = min(int(ratio), n_b)
        else:
            num_samples = max(1, int(np.ceil(ratio * n_b)))

        pts_b = x_np[idx_b]
        sampled = []

        start_idx = np.random.randint(n_b) if random_start else 0
        sampled.append(start_idx)
        min_dist = np.sum((pts_b - pts_b[start_idx]) ** 2, axis=-1)

        for _ in range(1, num_samples):
            next_idx = int(np.argmax(min_dist))
            sampled.append(next_idx)
            dist_new = np.sum((pts_b - pts_b[next_idx]) ** 2, axis=-1)
            min_dist = np.minimum(min_dist, dist_new)

        selected_indices.extend(idx_b[sampled])

    return ops.convert_to_tensor(np.array(selected_indices, dtype=np.int64), dtype="int64")


def radius(
    x,
    y,
    r: float,
    batch_x: Optional[any] = None,
    batch_y: Optional[any] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
):
    r"""Finds for each element in `y` all points in `x` within distance `r`."""
    x_np = ops.convert_to_numpy(x)
    y_np = ops.convert_to_numpy(y)
    if x_np.ndim == 1:
        x_np = x_np[:, None]
    if y_np.ndim == 1:
        y_np = y_np[:, None]

    N = x_np.shape[0]
    M = y_np.shape[0]

    if batch_x is None:
        batch_x_np = np.zeros(N, dtype=np.int64)
    else:
        batch_x_np = ops.convert_to_numpy(batch_x).astype(np.int64)

    if batch_y is None:
        batch_y_np = np.zeros(M, dtype=np.int64)
    else:
        batch_y_np = ops.convert_to_numpy(batch_y).astype(np.int64)

    rows = []
    cols = []
    r_sq = r * r

    for i in range(M):
        valid_b = batch_x_np == batch_y_np[i]
        valid_idx = np.where(valid_b)[0]
        if len(valid_idx) == 0:
            continue

        diff = x_np[valid_idx] - y_np[i]
        dist_sq = np.sum(diff**2, axis=-1)
        within_r = np.where(dist_sq <= r_sq)[0]
        if len(within_r) > max_num_neighbors:
            # Sort and take top max_num_neighbors
            sort_order = np.argsort(dist_sq[within_r])[:max_num_neighbors]
            within_r = within_r[sort_order]

        for match_idx in valid_idx[within_r]:
            rows.append(i)
            cols.append(match_idx)

    if len(rows) == 0:
        return ops.zeros((2, 0), dtype="int64")

    return ops.convert_to_tensor(np.stack([rows, cols], axis=0), dtype="int64")


def radius_graph(
    x,
    r: float,
    batch: Optional[any] = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
    batch_size: Optional[int] = None,
):
    r"""Computes graph edges to all points within a given distance `r`."""
    assert flow in ["source_to_target", "target_to_source"]
    edge_index = radius(
        x,
        x,
        r,
        batch_x=batch,
        batch_y=batch,
        max_num_neighbors=max_num_neighbors if loop else max_num_neighbors + 1,
    )
    edge_index_np = ops.convert_to_numpy(edge_index)

    if not loop and edge_index_np.shape[1] > 0:
        mask = edge_index_np[0] != edge_index_np[1]
        edge_index_np = edge_index_np[:, mask]

    if flow == "source_to_target" and edge_index_np.shape[1] > 0:
        edge_index_np = np.flip(edge_index_np, axis=0)

    return ops.convert_to_tensor(edge_index_np, dtype=edge_index.dtype)


def nearest(
    x,
    y,
    batch_x: Optional[any] = None,
    batch_y: Optional[any] = None,
):
    r"""Clusters each point in `x` to its nearest point in `y`."""
    edge_index = knn(y, x, k=1, batch_x=batch_y, batch_y=batch_x)
    return edge_index[1]

