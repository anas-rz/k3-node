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
    from k3_node.layers.conv.utils import is_tracing

    if not is_tracing(index):
        try:
            # `index` is purely structural (never differentiated), so plain
            # numpy is fine for it. `x` itself is placed into the dense
            # tensor with the differentiable `ops.scatter` below -- a numpy
            # round-trip on `x` would silently detach it from the graph and
            # stop gradients from flowing back into whatever produced it.
            index_np = ops.convert_to_numpy(index).astype(np.int64)
            N = len(index_np)

            B = int(np.max(index_np)) + 1 if N > 0 else 0
            if dim_size is not None:
                B = max(B, int(dim_size))

            # Compute local index for each node in its graph. `index` is
            # required to be sorted (see `assert_sorted_index`), so the
            # first occurrence of each value is its graph's start offset.
            counts = np.bincount(index_np, minlength=B)
            max_nodes = int(np.max(counts)) if len(counts) > 0 else 0
            if max_num_elements is not None:
                max_nodes = max(max_nodes, int(max_num_elements))

            local_index = np.arange(N) - np.searchsorted(index_np, index_np, side="left")
            valid = local_index < max_nodes

            mask_np = np.zeros((B, max_nodes), dtype=bool)
            mask_np[index_np[valid], local_index[valid]] = True

            feat_shape = tuple(ops.shape(x)[1:])
            scatter_idx = np.stack([index_np[valid], local_index[valid]], axis=1)
            x_valid = x if bool(valid.all()) else ops.take(x, np.nonzero(valid)[0], axis=0)
            out = ops.scatter(scatter_idx, x_valid, shape=(B, max_nodes, *feat_shape))

            if fill_value != 0.0:
                mask_t = ops.convert_to_tensor(mask_np)
                mask_expanded = ops.reshape(mask_t, (B, max_nodes) + (1,) * len(feat_shape))
                fill = ops.full((B, max_nodes, *feat_shape), fill_value, dtype=x.dtype)
                out = ops.where(mask_expanded, out, fill)

            return out, ops.convert_to_tensor(mask_np, dtype="bool")
        except Exception:
            pass

    # Pure ops implementation for symbolic tracing / graph execution
    N = ops.shape(x)[0]
    idx_col = ops.expand_dims(index, 1)
    idx_row = ops.expand_dims(index, 0)
    same_seg = ops.cast(ops.equal(idx_col, idx_row), "int32")
    tril = ops.tril(ops.ones((N, N), dtype="int32"))
    local_idx = ops.sum(same_seg * tril, axis=1) - 1

    if max_num_elements is not None:
        max_nodes = int(max_num_elements)
    elif hasattr(x, "shape") and x.shape[0] is not None:
        max_nodes = int(x.shape[0])
    else:
        max_nodes = ops.max(local_idx) + 1

    if dim_size is not None and isinstance(dim_size, int):
        B = dim_size
    elif hasattr(index, "shape") and index.shape[0] is not None and dim_size is not None:
        B = int(dim_size)
    else:
        B = ops.max(index) + 1

    dense_x = ops.full((B, max_nodes, *ops.shape(x)[1:]), fill_value, dtype=x.dtype)
    mask = ops.zeros((B, max_nodes), dtype="bool")

    scatter_indices = ops.stack([ops.cast(index, "int32"), ops.cast(local_idx, "int32")], axis=1)
    dense_x = ops.scatter_update(dense_x, scatter_indices, x)
    mask = ops.scatter_update(mask, scatter_indices, ops.ones((N,), dtype="bool"))
    return dense_x, mask


class Aggregation(layers.Layer):
    r"""An abstract base class for implementing custom aggregations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.built = True

    def build(self, input_shape=None):
        self.built = True

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
        dim_total = len(x.shape) if hasattr(x, "shape") and x.shape is not None else len(ops.shape(x))
        if dim >= dim_total or dim < -dim_total:
            raise ValueError(
                f"Encountered invalid dimension '{dim}' of source tensor with "
                f"{dim_total} dimensions"
            )

        if index is None and ptr is None:
            N = x.shape[dim] if hasattr(x, "shape") and x.shape[dim] is not None else ops.shape(x)[dim]
            index = ops.zeros((N,), dtype="int32")

        if ptr is not None and index is None:
            index = ptr2index(ptr)

        if ptr is not None:
            ptr_len = ptr.shape[0] if hasattr(ptr, "shape") and ptr.shape[0] is not None else ops.shape(ptr)[0]
            if dim_size is None:
                dim_size = ptr_len - 1
            elif dim_size != ptr_len - 1:
                raise ValueError(
                    f"Encountered invalid 'dim_size' (got '{dim_size}' but "
                    f"expected '{ptr_len - 1}')"
                )

        if index is not None and dim_size is None:
            dim_size = ops.max(index) + 1
            try:
                dim_size = int(dim_size)
            except Exception:
                pass

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
            from k3_node.layers.conv.utils import is_tracing
            if is_tracing(index):
                return
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
            val = ops.segment_max(x, index, num_segments=dim_size)
            ones = ops.ones_like(x)
            count = ops.segment_sum(ones, index, num_segments=dim_size)
            return ops.where(ops.greater(count, 0), val, ops.zeros_like(val))
        elif reduce == "min":
            val = -ops.segment_max(-x, index, num_segments=dim_size)
            ones = ops.ones_like(x)
            count = ops.segment_sum(ones, index, num_segments=dim_size)
            return ops.where(ops.greater(count, 0), val, ops.zeros_like(val))
        elif reduce == "mul":
            log_abs = ops.log(ops.maximum(ops.abs(x), 1e-7))
            sum_log = ops.segment_sum(log_abs, index, num_segments=dim_size)
            neg_count = ops.segment_sum(ops.cast(ops.less(x, 0.0), dtype=x.dtype), index, num_segments=dim_size)
            sign = ops.cos(ops.cast(3.141592653589793, dtype=x.dtype) * neg_count)
            return ops.exp(sum_log) * sign
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
