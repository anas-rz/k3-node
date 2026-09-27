from typing import Optional
from keras import ops
import numpy as np

from .knn import knn
from k3_node.ops.host import to_numpy


def approx_knn(
    x,
    y,
    k: int,
    batch_x: Optional[any] = None,
    batch_y: Optional[any] = None,
):
    r"""Finds for each element in `y` the `k` approximated nearest points in `x`.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import approx_knn

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        pos = np.random.rand(10, 3).astype("float32")  # 3D positions
        query = np.random.rand(4, 3).astype("float32")  # 4 query points

        assign = approx_knn(pos, query, k=3)  # approximate (NN-descent) nearest neighbors
        print(tuple(assign.shape))  # (2, 12)
        ```
    """
    try:
        from pynndescent import NNDescent

        if batch_x is None:
            batch_x = ops.zeros((ops.shape(x)[0],), dtype="int32")
        if batch_y is None:
            batch_y = ops.zeros((ops.shape(y)[0],), dtype="int32")

        x_np = to_numpy(x)
        y_np = to_numpy(y)
        batch_x_np = to_numpy(batch_x)
        batch_y_np = to_numpy(batch_y)

        min_xy = min(np.min(x_np), np.min(y_np))
        x_np = x_np - min_xy
        y_np = y_np - min_xy

        max_xy = max(np.max(x_np), np.max(y_np))
        if max_xy > 0:
            x_np = x_np / max_xy
            y_np = y_np / max_xy

        x_aug = np.concatenate([x_np, 2.0 * x_np.shape[1] * batch_x_np[:, None]], axis=-1)
        y_aug = np.concatenate([y_np, 2.0 * y_np.shape[1] * batch_y_np[:, None]], axis=-1)

        index = NNDescent(x_aug)
        col, dist = index.query(y_aug, k=k)
        row = np.repeat(np.arange(y_np.shape[0]), k)
        col = col.reshape(-1)
        dist = dist.reshape(-1)

        valid = ~np.isinf(dist)
        return ops.convert_to_tensor(np.stack([row[valid], col[valid]], axis=0), dtype="int64")
    except ImportError:
        return knn(x, y, k, batch_x=batch_x, batch_y=batch_y)


def approx_knn_graph(
    x,
    k: int,
    batch: Optional[any] = None,
    loop: bool = False,
    flow: str = "source_to_target",
):
    r"""Computes graph edges to the nearest approximated `k` points.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import approx_knn_graph

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        pos = np.random.rand(10, 3).astype("float32")  # 3D positions

        edge_index = approx_knn_graph(pos, k=3)
        print(edge_index.shape[0])  # 2
        ```
    """
    assert flow in ["source_to_target", "target_to_source"]
    edge_index = approx_knn(x, x, k if loop else k + 1, batch, batch)
    edge_index_np = to_numpy(edge_index)

    if flow == "source_to_target":
        edge_index_np = np.flip(edge_index_np, axis=0)

    if not loop:
        mask = edge_index_np[0] != edge_index_np[1]
        edge_index_np = edge_index_np[:, mask]

    return ops.convert_to_tensor(edge_index_np, dtype=edge_index.dtype)

