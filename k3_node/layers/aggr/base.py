from typing import Optional, Tuple
from keras import layers, ops
import numpy as np
from k3_node.ops.segment import segment_max, segment_sum
from k3_node.ops.creation import scatter, full


def ptr2index(ptr):
    r"""Converts a pointer tensor into an index tensor.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import ptr2index

        ptr = np.array([0, 3, 5])  # CSR pointer: set 0 has 3 elements, set 1 has 2
        print(tuple(ptr2index(ptr).shape))  # (5,): set id of every element
        ```
    """
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

    Example:
        ```python
        import numpy as np
        from k3_node.layers import to_dense_batch

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        batch = np.repeat([0, 1], 5)  # nodes 0-4 belong to graph 0, nodes 5-9 to graph 1

        x_dense, mask = to_dense_batch(x, batch)  # [num_graphs, max_nodes, features] + validity mask
        print(tuple(x_dense.shape), tuple(mask.shape))  # (2, 5, 8) (2, 5)
        ```
    """
    from k3_node.layers.conv.utils import is_tracing

    if not is_tracing(index):
        try:
            # `index` is purely structural (never differentiated), so plain
            # numpy is fine for it. `x` itself is placed into the dense
            # tensor with the differentiable `ops.scatter` below -- a numpy
            # round-trip on `x` would silently detach it from the graph and
            # stop gradients from flowing back into whatever produced it.
            from k3_node.ops.host import to_numpy  # zeros during Keras' shape inference

            index_np = np.asarray(to_numpy(index)).astype(np.int64)
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
            out = scatter(scatter_idx, x_valid, shape=(B, max_nodes, *feat_shape))

            if fill_value != 0.0:
                mask_t = ops.convert_to_tensor(mask_np)
                mask_expanded = ops.reshape(mask_t, (B, max_nodes) + (1,) * len(feat_shape))
                fill = full((B, max_nodes, *feat_shape), fill_value, dtype=x.dtype)
                out = ops.where(mask_expanded, out, fill)

            return out, ops.convert_to_tensor(mask_np, dtype="bool")
        except Exception:
            pass

    # Pure ops implementation for symbolic tracing / graph execution. `index` is sorted, so a
    # node's position in its graph is its offset from the graph's first node (O(N) memory).
    from k3_node.ops.segment import segment_sum

    N = ops.shape(x)[0]
    index = ops.cast(index, "int32")
    num_segments = int(dim_size) if dim_size is not None else N
    counts = segment_sum(ops.ones_like(index), index, num_segments=num_segments)
    starts = ops.cumsum(counts) - counts
    local_idx = ops.arange(N, dtype="int32") - ops.take(starts, index, axis=0)

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

    dense_x = full((B, max_nodes, *ops.shape(x)[1:]), fill_value, dtype=x.dtype)
    mask = ops.zeros((B, max_nodes), dtype="bool")

    scatter_indices = ops.stack([ops.cast(index, "int32"), ops.cast(local_idx, "int32")], axis=1)
    dense_x = ops.scatter_update(dense_x, scatter_indices, x)
    mask = ops.scatter_update(mask, scatter_indices, ops.ones((N,), dtype="bool"))
    return dense_x, mask


def from_dense_batch(x_dense, index):
    r"""The inverse of :func:`to_dense_batch`: turns ``[batch_size, max_nodes, *dims]`` back into
    one row per node, in the order given by the sorted ``index``. Static shapes, differentiable.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import from_dense_batch, to_dense_batch

        x = np.random.rand(10, 8).astype("float32")  # 10 nodes with 8 features each
        batch = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])  # three graphs of different sizes

        x_dense, mask = to_dense_batch(x, batch)
        print(np.allclose(from_dense_batch(x_dense, batch), x))  # True
        ```
    """
    from k3_node.ops.segment import segment_sum

    index = ops.cast(ops.convert_to_tensor(index), "int32")
    num_graphs, max_nodes = ops.shape(x_dense)[0], ops.shape(x_dense)[1]
    counts = segment_sum(ops.ones_like(index), index, num_segments=num_graphs)
    starts = ops.cumsum(counts) - counts
    position = ops.arange(ops.shape(index)[0], dtype="int32") - ops.take(starts, index, axis=0)
    flat = ops.reshape(x_dense, (-1,) + tuple(x_dense.shape[2:]))
    return ops.take(flat, index * max_nodes + position, axis=0)


def to_dense_adj(edge_index, batch=None, edge_attr=None, max_num_nodes: Optional[int] = None,
                 batch_size: Optional[int] = None):
    r"""Converts a batch of graphs into dense adjacency matrices of shape
    ``[num_graphs, max_nodes, max_nodes]`` (or ``[..., edge_features]`` with ``edge_attr``), as in PyG.

    The node dimension matches :func:`to_dense_batch` for the same ``batch`` vector, so the two can
    be used together. Duplicate edges are summed.

    Example:
        ```python
        import numpy as np
        from k3_node.layers import to_dense_adj

        edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 0, 4, 3]])
        batch = np.array([0, 0, 0, 1, 1])  # two graphs with 3 and 2 nodes
        adj = to_dense_adj(edge_index, batch)
        print(tuple(adj.shape))  # (2, 3, 3)
        ```
    """
    from k3_node.layers.conv.utils import is_tracing

    edge_index = ops.cast(ops.convert_to_tensor(edge_index), "int32")
    if batch is None:
        num_nodes = max_num_nodes or (int(ops.convert_to_numpy(ops.max(edge_index))) + 1)
        batch = ops.zeros((num_nodes,), dtype="int32")
    batch = ops.cast(ops.convert_to_tensor(batch), "int32")

    if not is_tracing(batch):
        index_np = ops.convert_to_numpy(batch).astype(np.int64)
        num_graphs = int(index_np.max()) + 1 if len(index_np) else 0
        num_graphs = max(num_graphs, int(batch_size or 0))
        local = np.arange(len(index_np)) - np.searchsorted(index_np, index_np, side="left")
        max_nodes = int(np.bincount(index_np, minlength=num_graphs).max()) if len(index_np) else 0
        max_nodes = max(max_nodes, int(max_num_nodes or 0))
        local = ops.convert_to_tensor(local.astype("int32"))
    else:  # compiled: same local indices as to_dense_batch, padded to the total node count
        n = ops.shape(batch)[0]
        same = ops.cast(ops.equal(ops.expand_dims(batch, 1), ops.expand_dims(batch, 0)), "int32")
        local = ops.sum(same * ops.tril(ops.ones((n, n), dtype="int32")), axis=1) - 1
        num_graphs = batch_size if batch_size is not None else ops.max(batch) + 1
        max_nodes = max_num_nodes if max_num_nodes is not None else batch.shape[0]

    src, dst = edge_index[0], edge_index[1]
    indices = ops.stack([ops.take(batch, src), ops.take(local, src), ops.take(local, dst)], axis=1)
    num_edges = ops.shape(edge_index)[1]
    values = ops.ones((num_edges,), dtype="float32") if edge_attr is None else ops.convert_to_tensor(edge_attr)
    extra = tuple(values.shape[1:])
    return scatter(indices, values, (num_graphs, max_nodes, max_nodes) + extra)


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
        # Plain NumPy inputs cannot be mixed with backend tensors (e.g. `ndarray - torch.Tensor`).
        if isinstance(x, np.ndarray):
            x = ops.convert_to_tensor(x)
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
            return segment_sum(x, index, num_segments=dim_size)
        elif reduce == "mean":
            sum_val = segment_sum(x, index, num_segments=dim_size)
            ones = ops.ones_like(x)
            count = segment_sum(ones, index, num_segments=dim_size)
            return sum_val / ops.maximum(count, 1.0)
        elif reduce == "max":
            val = segment_max(x, index, num_segments=dim_size)
            ones = ops.ones_like(x)
            count = segment_sum(ones, index, num_segments=dim_size)
            return ops.where(ops.greater(count, 0), val, ops.zeros_like(val))
        elif reduce == "min":
            val = -segment_max(-x, index, num_segments=dim_size)
            ones = ops.ones_like(x)
            count = segment_sum(ones, index, num_segments=dim_size)
            return ops.where(ops.greater(count, 0), val, ops.zeros_like(val))
        elif reduce == "mul":
            log_abs = ops.log(ops.maximum(ops.abs(x), 1e-7))
            sum_log = segment_sum(log_abs, index, num_segments=dim_size)
            neg_count = segment_sum(ops.cast(ops.less(x, 0.0), dtype=x.dtype), index, num_segments=dim_size)
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
