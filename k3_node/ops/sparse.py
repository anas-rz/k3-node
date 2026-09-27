import keras
from keras import ops

from k3_node.ops.segment import segment_sum


def spmm(index_targets, index_sources, edge_weight, x, num_targets: int):
    r"""Sums weighted source features into targets:
    :math:`\mathbf{out}_i = \sum_{(j, i)} w_{ji} \, \mathbf{x}_j`, i.e. a sparse-dense matrix product.

    On torch this uses ``torch.sparse.mm``, which never materializes the ``[num_edges, channels]``
    per-edge messages (the dominant memory cost of wide GCN-style layers). Other backends gather and
    sum, which XLA fuses when compiled.

    Args:
        index_targets: Target node of every edge, shape ``[num_edges]``.
        index_sources: Source node of every edge, shape ``[num_edges]``.
        edge_weight: Edge weights of shape ``[num_edges]``, or ``None`` for weight 1.
        x: Source node features of shape ``[num_sources, channels]``.
        num_targets (int): The number of target nodes.

    Example:
        ```python
        import numpy as np
        from k3_node.ops.sparse import spmm

        x = np.array([[1.0], [2.0], [3.0]], dtype="float32")  # one feature per node
        targets, sources = np.array([0, 0, 2]), np.array([1, 2, 0])  # edges 1->0, 2->0, 0->2
        weight = np.array([1.0, 0.5, 2.0], dtype="float32")

        print(np.asarray(spmm(targets, sources, weight, x, 3)).ravel())  # [3.5 0.  2. ]
        ```
    """
    if keras.config.backend() == "torch":
        import torch

        x_t = ops.convert_to_tensor(x)
        if x_t.device.type != "meta" and len(x_t.shape) == 2:
            index = torch.stack([ops.convert_to_tensor(index_targets).long(),
                                 ops.convert_to_tensor(index_sources).long()])
            if edge_weight is None:
                values = torch.ones(index.shape[1], dtype=x_t.dtype, device=x_t.device)
            else:
                values = ops.convert_to_tensor(edge_weight).to(x_t.dtype).reshape(-1)
            adj = torch.sparse_coo_tensor(index, values, (int(num_targets), x_t.shape[0]), check_invariants=True)
            return torch.sparse.mm(adj, x_t)

    messages = ops.take(x, index_sources, axis=0)
    if edge_weight is not None:
        messages = ops.expand_dims(ops.cast(edge_weight, messages.dtype), -1) * messages
    return segment_sum(messages, ops.cast(index_targets, "int32"), num_segments=num_targets)
