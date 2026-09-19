from typing import List, Optional, Union
from keras import ops
import numpy as np


def voxel_grid(
    pos,
    size: Union[float, List[float], any],
    batch: Optional[any] = None,
    start: Optional[Union[float, List[float], any]] = None,
    end: Optional[Union[float, List[float], any]] = None,
):
    r"""Voxel grid pooling that clusters points within the same voxel."""
    pos_np = ops.convert_to_numpy(pos)
    if pos_np.ndim == 1:
        pos_np = pos_np[:, None]
    dim = pos_np.shape[1]

    if isinstance(size, (int, float)):
        size_np = np.full(dim, float(size))
    else:
        size_np = np.array(size, dtype=np.float64)

    if start is None:
        start_np = np.min(pos_np, axis=0)
    elif isinstance(start, (int, float)):
        start_np = np.full(dim, float(start))
    else:
        start_np = np.array(start, dtype=np.float64)

    if end is None:
        end_np = np.max(pos_np, axis=0)
    elif isinstance(end, (int, float)):
        end_np = np.full(dim, float(end))
    else:
        end_np = np.array(end, dtype=np.float64)

    # Number of bins per dimension
    num_bins = np.floor((end_np - start_np) / size_np).astype(np.int64) + 1

    # Discretized coordinates
    c = np.floor((pos_np - start_np) / size_np).astype(np.int64)

    # Compute flat index
    cluster_np = np.zeros(len(pos_np), dtype=np.int64)
    multiplier = 1
    for d in range(dim):
        cluster_np += c[:, d] * multiplier
        multiplier *= num_bins[d]

    if batch is not None:
        batch_np = ops.convert_to_numpy(batch).astype(np.int64)
        cluster_np += batch_np * multiplier

    return ops.convert_to_tensor(cluster_np, dtype="int64")

