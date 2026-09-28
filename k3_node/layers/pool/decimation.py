from typing import Tuple, Union
from keras import ops
import numpy as np
from k3_node.ops.host import _in_shape_inference, to_numpy


def decimation_indices(
    ptr,
    decimation_factor: Union[int, float],
) -> Tuple[any, any]:
    r"""Gets indices which downsample each point cloud by a decimation factor.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import decimation_indices

        ptr = np.array([0, 4, 10])  # two graphs with 4 and 6 nodes
        index, new_ptr = decimation_indices(ptr, decimation_factor=2)  # keep every 2nd node per graph
        print(tuple(index.shape), tuple(new_ptr.shape))  # (5,) (3,)
        ```
    """
    if decimation_factor < 1:
        raise ValueError(
            f"The argument `decimation_factor` should be higher than (or "
            f"equal to) 1 for downsampling. (got {decimation_factor})"
        )

    ptr_np = to_numpy(ptr)
    batch_size = len(ptr_np) - 1
    count = ptr_np[1:] - ptr_np[:-1]
    if _in_shape_inference():  # `ptr` is placeholder zeros: keep one (valid) node per graph
        count = np.maximum(count, 1)
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

