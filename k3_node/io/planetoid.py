import os.path as osp
import pickle
import warnings
from typing import Dict, List, Optional
import numpy as np
from keras import ops

from k3_node.data import Data
from k3_node.io.txt_array import read_txt_array
from k3_node.layers.conv.utils import remove_self_loops
from k3_node.utils.graph import coalesce


def index_to_mask(index, size: int):
    """Converts 1D index array into a boolean mask."""
    mask = np.zeros(size, dtype=bool)
    idx_np = ops.convert_to_numpy(index)
    mask[idx_np] = True
    return ops.convert_to_tensor(mask, dtype="bool")


def edge_index_from_dict(graph_dict: Dict[int, List[int]], num_nodes: Optional[int] = None):
    rows: List[int] = []
    cols: List[int] = []
    for key, val_list in graph_dict.items():
        for val in val_list:
            rows.append(key)
            cols.append(val)
    if len(rows) == 0:
        return ops.zeros((2, 0), dtype="int64")

    edge_index = np.array([rows, cols], dtype=np.int64)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = coalesce(edge_index, num_nodes=num_nodes, sort_by_row=False)
    return ops.convert_to_tensor(edge_index, dtype="int64")


def read_file(folder: str, prefix: str, name: str):
    path = osp.join(folder, f"ind.{prefix.lower()}.{name}")
    if name == "test.index":
        return read_txt_array(path, dtype="int64")

    with open(path, "rb") as f:
        warnings.filterwarnings("ignore", ".*`scipy.sparse.csr` name.*")
        out = pickle.load(f, encoding="latin1")

    if name == "graph":
        return out

    if hasattr(out, "todense"):
        out = out.todense()
    return np.array(out, dtype=np.float32)


def read_planetoid_data(folder: str, prefix: str) -> Data:
    """Reads planetoid citation graph files and returns a `Data` object."""
    names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items

    test_index_np = ops.convert_to_numpy(test_index).astype(np.int64)
    sorted_test_index = np.sort(test_index_np)

    train_index = np.arange(y.shape[0], dtype=np.int64)
    val_index = np.arange(y.shape[0], y.shape[0] + 500, dtype=np.int64)

    if prefix.lower() == "citeseer":
        len_test_indices = int(np.max(test_index_np) - np.min(test_index_np)) + 1
        tx_ext = np.zeros((len_test_indices, tx.shape[1]), dtype=tx.dtype)
        tx_ext[sorted_test_index - np.min(test_index_np), :] = tx
        ty_ext = np.zeros((len_test_indices, ty.shape[1]), dtype=ty.dtype)
        ty_ext[sorted_test_index - np.min(test_index_np), :] = ty
        tx, ty = tx_ext, ty_ext

    x = np.concatenate([allx, tx], axis=0)
    x[test_index_np] = x[sorted_test_index]

    y_cat = np.concatenate([ally, ty], axis=0)
    y = np.argmax(y_cat, axis=1).astype(np.int64)
    y[test_index_np] = y[sorted_test_index]

    num_nodes = y.shape[0]
    train_mask = index_to_mask(train_index, size=num_nodes)
    val_mask = index_to_mask(val_index, size=num_nodes)
    test_mask = index_to_mask(test_index_np, size=num_nodes)

    edge_index = edge_index_from_dict(graph, num_nodes=num_nodes)

    data = Data(
        x=ops.convert_to_tensor(x, dtype="float32"),
        edge_index=edge_index,
        y=ops.convert_to_tensor(y, dtype="int64"),
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
