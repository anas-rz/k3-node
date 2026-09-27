from typing import Tuple
from keras import ops
import numpy as np


def consecutive_cluster(src) -> Tuple[any, any]:
    r"""Maps elements in `src` to consecutive integers starting from 0,
    returning the mapped indices and a permutation array of representative indices.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import consecutive_cluster

        cluster = np.array([4, 4, 9, 2, 9])
        new_cluster, perm = consecutive_cluster(cluster)  # relabel cluster ids to 0..num_clusters-1
        print(tuple(new_cluster.shape), tuple(perm.shape))  # (5,) (3,)
        ```
    """
    src_np = ops.convert_to_numpy(src)
    unique, inv = np.unique(src_np, return_inverse=True)
    perm = np.empty(len(unique), dtype=inv.dtype)
    arange = np.arange(len(inv), dtype=inv.dtype)
    perm[inv] = arange

    inv_tensor = ops.convert_to_tensor(inv, dtype=src.dtype)
    perm_tensor = ops.convert_to_tensor(perm, dtype=src.dtype)
    return inv_tensor, perm_tensor

