from typing import Any, Dict
import numpy as np
import scipy.sparse as sp
from keras import ops

from k3_node.data import Data
from k3_node.layers.conv.utils import remove_self_loops
from k3_node.transforms.utils import to_undirected as to_undirected_fn


def read_npz(path: str, to_undirected: bool = True) -> Data:
    """Reads a `.npz` graph archive (e.g. Amazon, Coauthor) and returns a `Data` object."""
    with np.load(path) as f:
        return parse_npz(f, to_undirected=to_undirected)


def parse_npz(f: Dict[str, Any], to_undirected: bool = True) -> Data:
    x_sp = sp.csr_matrix(
        (f["attr_data"], f["attr_indices"], f["attr_indptr"]),
        shape=tuple(f["attr_shape"]),
    )
    x = np.array(x_sp.todense(), dtype=np.float32)
    x[x > 0] = 1.0

    adj = sp.csr_matrix(
        (f["adj_data"], f["adj_indices"], f["adj_indptr"]),
        shape=tuple(f["adj_shape"]),
    ).tocoo()

    row = np.array(adj.row, dtype=np.int64)
    col = np.array(adj.col, dtype=np.int64)
    edge_index = np.stack([row, col], axis=0)

    edge_index, _ = remove_self_loops(edge_index)
    if to_undirected:
        edge_index = to_undirected_fn(edge_index, num_nodes=x.shape[0])

    y = np.array(f["labels"], dtype=np.int64)

    return Data(
        x=ops.convert_to_tensor(x, dtype="float32"),
        edge_index=ops.convert_to_tensor(edge_index, dtype="int64"),
        y=ops.convert_to_tensor(y, dtype="int64"),
    )

