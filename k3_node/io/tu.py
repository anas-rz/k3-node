import glob
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from keras import ops

from k3_node.data import Data
from k3_node.io.txt_array import read_txt_array
from k3_node.layers.conv.utils import remove_self_loops
from k3_node.utils.graph import coalesce


def np_one_hot(arr: np.ndarray) -> np.ndarray:
    """One-hot encodes a 1D integer array."""
    arr = arr.astype(np.int64)
    if arr.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    min_val = np.min(arr)
    arr = arr - min_val
    max_val = np.max(arr)
    out = np.zeros((arr.shape[0], max_val + 1), dtype=np.float32)
    out[np.arange(arr.shape[0]), arr] = 1.0
    return out


def read_tu_data(folder: str, prefix: str) -> Tuple[Data, Dict[str, Any], Dict[str, int]]:
    """Reads a TU benchmark dataset from disk and returns `(data, slices, sizes)`."""
    files = sorted(glob.glob(osp.join(folder, f"{prefix}_*.txt")))
    names = [osp.basename(f)[len(prefix) + 1 : -4] for f in files]

    def read_file(name: str, dtype: str = "float32"):
        path = osp.join(folder, f"{prefix}_{name}.txt")
        return ops.convert_to_numpy(read_txt_array(path, sep=",", dtype=dtype))

    # Adjacency: 1-indexed in raw TU files
    edge_index_np = read_file("A", dtype="int64")
    if edge_index_np.ndim == 1:
        edge_index_np = edge_index_np.reshape(-1, 2)
    edge_index_np = edge_index_np.T - 1  # 0-indexed (2, E)

    # Graph indicator: which graph each node belongs to (1-indexed)
    batch_np = read_file("graph_indicator", dtype="int64") - 1  # 0-indexed (N,)

    node_attribute = np.empty((batch_np.shape[0], 0), dtype=np.float32)
    if "node_attributes" in names:
        node_attribute = read_file("node_attributes", dtype="float32")
        if node_attribute.ndim == 1:
            node_attribute = node_attribute[:, None]

    node_label = np.empty((batch_np.shape[0], 0), dtype=np.float32)
    if "node_labels" in names:
        node_label_raw = read_file("node_labels", dtype="int64")
        if node_label_raw.ndim == 1:
            node_label_raw = node_label_raw[:, None]
        encoded = [np_one_hot(node_label_raw[:, i]) for i in range(node_label_raw.shape[1])]
        node_label = np.concatenate(encoded, axis=-1)

    edge_attribute = np.empty((edge_index_np.shape[1], 0), dtype=np.float32)
    if "edge_attributes" in names:
        edge_attribute = read_file("edge_attributes", dtype="float32")
        if edge_attribute.ndim == 1:
            edge_attribute = edge_attribute[:, None]

    edge_label = np.empty((edge_index_np.shape[1], 0), dtype=np.float32)
    if "edge_labels" in names:
        edge_label_raw = read_file("edge_labels", dtype="int64")
        if edge_label_raw.ndim == 1:
            edge_label_raw = edge_label_raw[:, None]
        encoded = [np_one_hot(edge_label_raw[:, i]) for i in range(edge_label_raw.shape[1])]
        edge_label = np.concatenate(encoded, axis=-1)

    # Combine attributes and one-hot labels
    x_list = [arr for arr in [node_attribute, node_label] if arr.shape[1] > 0]
    x_np = np.concatenate(x_list, axis=-1) if len(x_list) > 0 else None

    edge_attr_list = [arr for arr in [edge_attribute, edge_label] if arr.shape[1] > 0]
    edge_attr_np = np.concatenate(edge_attr_list, axis=-1) if len(edge_attr_list) > 0 else None

    y_np = None
    if "graph_attributes" in names:
        y_np = read_file("graph_attributes", dtype="float32")
    elif "graph_labels" in names:
        y_raw = read_file("graph_labels", dtype="int64")
        _, y_inv = np.unique(y_raw, return_inverse=True)
        y_np = y_inv.astype(np.int64)

    num_nodes = x_np.shape[0] if x_np is not None else int(np.max(edge_index_np)) + 1
    edge_index_np, edge_attr_np = remove_self_loops(edge_index_np, edge_attr_np)
    edge_index_t, edge_attr_t = coalesce(edge_index_np, edge_attr_np, num_nodes=num_nodes)
    edge_index_np = ops.convert_to_numpy(edge_index_t)
    edge_attr_np = ops.convert_to_numpy(edge_attr_t) if edge_attr_t is not None else None

    # Convert to tensors
    edge_index = ops.convert_to_tensor(edge_index_np, dtype="int64")
    x = ops.convert_to_tensor(x_np, dtype="float32") if x_np is not None else None
    edge_attr = ops.convert_to_tensor(edge_attr_np, dtype="float32") if edge_attr_np is not None else None
    y = ops.convert_to_tensor(y_np, dtype="float32" if "graph_attributes" in names else "int64") if y_np is not None else None

    # Compute graph slices
    num_graphs = int(np.max(batch_np)) + 1
    node_counts = np.bincount(batch_np, minlength=num_graphs)
    node_slice = np.pad(np.cumsum(node_counts), (1, 0))

    row = edge_index_np[0]
    edge_batch = batch_np[row]
    edge_counts = np.bincount(edge_batch, minlength=num_graphs)
    edge_slice = np.pad(np.cumsum(edge_counts), (1, 0))

    # Shift edge indices so each graph starts at 0
    shift = node_slice[edge_batch]
    edge_index_shifted = edge_index_np - shift[None, :]
    edge_index = ops.convert_to_tensor(edge_index_shifted, dtype="int64")

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    slices: Dict[str, Any] = {
        "edge_index": ops.convert_to_tensor(edge_slice, dtype="int64"),
    }
    if x is not None:
        slices["x"] = ops.convert_to_tensor(node_slice, dtype="int64")
    else:
        data.num_nodes = int(batch_np.shape[0])
    if edge_attr is not None:
        slices["edge_attr"] = ops.convert_to_tensor(edge_slice, dtype="int64")
    if y is not None:
        if y_np.shape[0] == batch_np.shape[0]:
            slices["y"] = ops.convert_to_tensor(node_slice, dtype="int64")
        else:
            slices["y"] = ops.convert_to_tensor(np.arange(num_graphs + 1, dtype=np.int64), dtype="int64")

    sizes = {
        "num_node_attributes": node_attribute.shape[-1],
        "num_node_labels": node_label.shape[-1],
        "num_edge_attributes": edge_attribute.shape[-1],
        "num_edge_labels": edge_label.shape[-1],
    }

    return data, slices, sizes
