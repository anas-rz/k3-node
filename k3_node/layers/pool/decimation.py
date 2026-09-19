from typing import Tuple, Union
from keras import ops
import numpy as np


def decimation_indices(
    ptr,
    decimation_factor: Union[int, float],
) -> Tuple[any, any]:
    r"""Gets indices which downsample each point cloud by a decimation factor."""
    if decimation_factor < 1:
        raise ValueError(
            f"The argument `decimation_factor` should be higher than (or "
            f"equal to) 1 for downsampling. (got {decimation_factor})"
        )

    ptr_np = ops.convert_to_numpy(ptr)
    batch_size = len(ptr_np) - 1
    count = ptr_np[1:] - ptr_np[:-1]
    decim_count = np.maximum(count // int(decimation_factor), 1)

    decim_indices_list = []
    for i in range(batch_size):
        perm = np.random.permutation(count[i])[:decim_count[i]]
        decim_indices_list.append(ptr_np[i] + perm)

    decim_indices = np.concatenate(decim_indices_list, axis=0)
    decim_ptr = np.concatenate([[0], np.cumsum(decim_count)])

    return (
        ops.convert_to_tensor(decim_indices, dtype=ptr.dtype),
        ops.convert_to_tensor(decim_ptr, dtype=ptr.dtype),
    )

