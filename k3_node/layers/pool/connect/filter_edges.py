from typing import Optional, Tuple
from keras import ops
import numpy as np

from .base import Connect, ConnectOutput
from ..select.base import SelectOutput


def filter_adj(
    edge_index,
    edge_attr: Optional[any] = None,
    node_index=None,
    cluster_index: Optional[any] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[any, Optional[any]]:
    r"""Filters out edges if their incident nodes are not in any cluster."""
    if num_nodes is None:
        num_nodes = int(ops.max(edge_index)) + 1 if ops.shape(edge_index)[1] > 0 else 0

    node_idx_np = ops.convert_to_numpy(node_index).astype(np.int64)
    if cluster_index is None:
        cluster_idx_np = np.arange(len(node_idx_np), dtype=np.int64)
    else:
        cluster_idx_np = ops.convert_to_numpy(cluster_index).astype(np.int64)

    mapping = np.full((num_nodes,), -1, dtype=np.int64)
    mapping[node_idx_np] = cluster_idx_np

    edge_index_np = ops.convert_to_numpy(edge_index).astype(np.int64)
    row = mapping[edge_index_np[0]]
    col = mapping[edge_index_np[1]]
    valid = (row >= 0) & (col >= 0)

    row = row[valid]
    col = col[valid]

    new_edge_index = ops.convert_to_tensor(np.stack([row, col], axis=0), dtype=edge_index.dtype)

    new_edge_attr = None
    if edge_attr is not None:
        edge_attr_np = ops.convert_to_numpy(edge_attr)[valid]
        new_edge_attr = ops.convert_to_tensor(edge_attr_np, dtype=edge_attr.dtype)

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

