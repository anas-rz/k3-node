from typing import Optional, Tuple
from keras import ops
import numpy as np

from .base import Connect, ConnectOutput
from ..select.base import SelectOutput


from k3_node.layers.conv.utils import is_tracing


def filter_adj(
    edge_index,
    edge_attr: Optional[any] = None,
    node_index=None,
    cluster_index: Optional[any] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[any, Optional[any]]:
    r"""Filters out edges if their incident nodes are not in any cluster."""
    if is_tracing(edge_index) or (node_index is not None and is_tracing(node_index)):
        return edge_index, edge_attr

    if node_index is None:
        return edge_index, edge_attr

    edge_index = ops.cast(edge_index, "int32")
    node_index = ops.cast(node_index, "int32")
    if cluster_index is None:
        cluster_index = ops.arange(ops.shape(node_index)[0], dtype="int32")
    else:
        cluster_index = ops.cast(cluster_index, "int32")

    if num_nodes is None:
        num_nodes = ops.max(node_index) + 1 if ops.shape(node_index)[0] > 0 else 0
        if ops.shape(edge_index)[1] > 0:
            num_nodes = ops.maximum(num_nodes, ops.max(edge_index) + 1)
    try:
        num_nodes = int(num_nodes)
    except (TypeError, ValueError):
        pass

    mapping = ops.full((num_nodes,), -1, dtype="int32")
    mapping = ops.scatter_update(mapping, ops.expand_dims(node_index, -1), cluster_index)

    row = ops.take(mapping, edge_index[0], axis=0)
    col = ops.take(mapping, edge_index[1], axis=0)
    mask = (row >= 0) & (col >= 0)
    valid_idx = ops.where(mask)
    if isinstance(valid_idx, (tuple, list)):
        valid_idx = valid_idx[0]
    valid_idx = ops.reshape(valid_idx, (-1,))

    new_edge_index = ops.stack(
        [ops.take(row, valid_idx, axis=0), ops.take(col, valid_idx, axis=0)], axis=0
    )

    new_edge_attr = None
    if edge_attr is not None:
        new_edge_attr = ops.take(edge_attr, valid_idx, axis=0)

    return new_edge_index, new_edge_attr


class FilterEdges(Connect):
    r"""Filters out edges if their incident nodes are not in any cluster."""
    def call(
        self,
        select_output: SelectOutput,
        edge_index,
        edge_attr: Optional[any] = None,
        batch: Optional[any] = None,
    ) -> ConnectOutput:
        new_edge_index, new_edge_attr = filter_adj(
            edge_index,
            edge_attr,
            select_output.node_index,
            select_output.cluster_index,
            num_nodes=select_output.num_nodes,
        )
        new_batch = self.get_pooled_batch(select_output, batch)
        return ConnectOutput(new_edge_index, new_edge_attr, new_batch)

