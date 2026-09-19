from typing import Optional, Tuple
from keras import layers, ops
import numpy as np


def ptr2index(ptr):
    r"""Converts a pointer tensor into an index tensor."""
    ptr_np = ops.convert_to_numpy(ptr).astype(np.int64)
    counts = ptr_np[1:] - ptr_np[:-1]
    index_np = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    return ops.convert_to_tensor(index_np, dtype="int32")


def to_dense_batch(
    x,
    index,
    dim_size: Optional[int] = None,
    fill_value: float = 0.0,
    max_num_elements: Optional[int] = None,
) -> Tuple[any, any]:
    r"""Transforms a batched feature tensor into a dense representation
    of shape `(batch_size, max_nodes, *dims)`.
    """
    x_np = ops.convert_to_numpy(x)
    index_np = ops.convert_to_numpy(index).astype(np.int64)
    N = len(index_np)

    B = int(np.max(index_np)) + 1 if len(index_np) > 0 else 0
    if dim_size is not None:
        B = max(B, dim_size)

    # Compute local index for each node in its graph
    counts = np.bincount(index_np, minlength=B)
    max_nodes = int(np.max(counts)) if len(counts) > 0 else 0
    if max_num_elements is not None:
        max_nodes = max(max_nodes, max_num_elements)

    out_np = np.full((B, max_nodes, *x_np.shape[1:]), fill_value, dtype=x_np.dtype)
    mask_np = np.zeros((B, max_nodes), dtype=bool)

    # Fast assignment
    curr_counts = np.zeros(B, dtype=np.int64)
    for i in range(N):
        b = index_np[i]
        pos = curr_counts[b]
        if pos < max_nodes:
            out_np[b, pos] = x_np[i]
            mask_np[b, pos] = True
        curr_counts[b] += 1

    return (
        ops.convert_to_tensor(out_np, dtype=x.dtype),
        ops.convert_to_tensor(mask_np, dtype="bool"),
    )


class Aggregation(layers.Layer):
    r"""An abstract base class for implementing custom aggregations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def __call__(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
        **kwargs,
    ):
        dim_total = len(ops.shape(x))
        if dim >= dim_total or dim < -dim_total:
            raise ValueError(
                f"Encountered invalid dimension '{dim}' of source tensor with "
                f"{dim_total} dimensions"
            )

        if index is None and ptr is None:
            N = ops.shape(x)[dim]
            index = ops.zeros((N,), dtype="int32")

        if ptr is not None and index is None:
            index = ptr2index(ptr)

        if ptr is not None:
            ptr_len = ops.shape(ptr)[0]
            if dim_size is None:
                dim_size = ptr_len - 1
            elif dim_size != ptr_len - 1:
                raise ValueError(
                    f"Encountered invalid 'dim_size' (got '{dim_size}' but "
                    f"expected '{ptr_len - 1}')"
                )

        if index is not None and dim_size is None:
            dim_size = int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0

        # Handle positional / keyword call to call()
        return self.call(
            x,
            index=index,
            ptr=ptr,
            dim_size=dim_size,
            dim=dim,
            max_num_elements=max_num_elements,
            **kwargs,
        )

    def call(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        max_num_elements: Optional[int] = None,
    ):
        raise NotImplementedError

    def assert_index_present(self, index: Optional[any]):
        if index is None:
            raise NotImplementedError("Aggregation requires 'index' to be specified")

    def assert_sorted_index(self, index: Optional[any]):
        if index is not None:
            idx_np = ops.convert_to_numpy(index)
            if not np.all(idx_np[:-1] <= idx_np[1:]):
                raise ValueError(
                    "Can not perform aggregation since the 'index' tensor is not sorted. "
                    "Specifically, if you use this aggregation as part of 'MessagePassing', "
                    "ensure that 'edge_index' is sorted by destination nodes."
                )

    def assert_two_dimensional_input(self, x, dim: int = -2):
        if len(ops.shape(x)) != 2:
            raise ValueError(
                f"Aggregation requires two-dimensional inputs (got '{len(ops.shape(x))}')"
            )
        if dim not in [-2, 0]:
            raise ValueError(
                f"Aggregation needs to perform aggregation in first dimension (got '{dim}')"
            )

    def reduce(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        reduce: str = "sum",
    ):
        r"""Reduces features along groups specified by `index` or `ptr`."""
        if ptr is not None and index is None:
            index = ptr2index(ptr)

        if index is None:
            raise RuntimeError("Aggregation requires 'index' to be specified")

        index = ops.cast(index, dtype="int32")
        if dim_size is None:
            dim_size = int(ops.max(index)) + 1 if ops.shape(index)[0] > 0 else 0

        if reduce in ["sum", "add"]:
            return ops.segment_sum(x, index, num_segments=dim_size)
        elif reduce == "mean":
            sum_val = ops.segment_sum(x, index, num_segments=dim_size)
            ones = ops.ones_like(x)
            count = ops.segment_sum(ones, index, num_segments=dim_size)
            return sum_val / ops.maximum(count, 1.0)
        elif reduce == "max":
            return ops.segment_max(x, index, num_segments=dim_size)
        elif reduce == "min":
            return -ops.segment_max(-x, index, num_segments=dim_size)
        elif reduce == "mul":
            x_np = ops.convert_to_numpy(x)
            idx_np = ops.convert_to_numpy(index).astype(np.int64)
            out_np = np.ones((dim_size, *x_np.shape[1:]), dtype=x_np.dtype)
            np.multiply.at(out_np, idx_np, x_np)
            return ops.convert_to_tensor(out_np, dtype=x.dtype)
        else:
            raise ValueError(f"Unsupported reduction '{reduce}'")

    def to_dense_batch(
        self,
        x,
        index: Optional[any] = None,
        ptr: Optional[any] = None,
        dim_size: Optional[int] = None,
        dim: int = -2,
        fill_value: float = 0.0,
        max_num_elements: Optional[int] = None,
    ) -> Tuple[any, any]:
        if ptr is not None and index is None:
            index = ptr2index(ptr)

        self.assert_index_present(index)
        self.assert_sorted_index(index)
        self.assert_two_dimensional_input(x, dim)

        return to_dense_batch(
            x,
            index,
            dim_size=dim_size,
            fill_value=fill_value,
            max_num_elements=max_num_elements,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
