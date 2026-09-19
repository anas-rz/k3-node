import copy
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
try:
    import torch
except ImportError:
    torch = None
    Tensor = type(None)

from k3_node.data import Data, HeteroData
from k3_node.data.storage import NodeStorage, EdgeStorage


def to_numpy_or_tensor(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, '__array__'):
        return np.asarray(x)
    return x


def index_select(value: Any, index: Any, dim: int = 0) -> Any:
    r"""Indexes the :obj:`value` tensor along dimension :obj:`dim` using the
    entries in :obj:`index`. Supports PyTorch, TensorFlow, JAX, and NumPy.
    """
    if torch is not None and isinstance(value, torch.Tensor):
        if not isinstance(index, torch.Tensor):
            index = torch.as_tensor(index, dtype=torch.long, device=value.device)
        else:
            index = index.to(dtype=torch.long, device=value.device)
        return torch.index_select(value, dim, index)

    # NumPy / Keras / JAX / TensorFlow array:
    if hasattr(value, 'numpy'):
        is_tf_or_jax = True
        np_val = value.numpy()
    elif isinstance(value, np.ndarray):
        is_tf_or_jax = False
        np_val = value
    elif hasattr(value, '__array__'):
        is_tf_or_jax = False
        np_val = np.asarray(value)
    else:
        return value

    if torch is not None and isinstance(index, torch.Tensor):
        np_idx = index.cpu().numpy()
    else:
        np_idx = np.asarray(index, dtype=np.int64)

    res = np.take(np_val, np_idx, axis=dim)
    if is_tf_or_jax:
        import keras
        return keras.ops.convert_to_tensor(res)
    return res


def filter_node_store_(store: NodeStorage, out_store: NodeStorage, index: Any):
    for key, value in store.items():
        if key == 'num_nodes':
            numel = index.numel() if hasattr(index, 'numel') else len(index)
            out_store.num_nodes = numel
        elif store.is_node_attr(key):
            dim = 0
            if hasattr(store, '_parent') and store._parent() is not None:
                dim = store._parent().__cat_dim__(key, value, store)
            out_store[key] = index_select(value, index, dim=dim)


def filter_edge_store_(
    store: EdgeStorage,
    out_store: EdgeStorage,
    row: Any,
    col: Any,
    index: Optional[Any],
    perm: Optional[Any] = None,
):
    for key, value in store.items():
        if key == 'edge_index':
            if torch is not None and isinstance(row, torch.Tensor):
                edge_index = torch.stack([row, col], dim=0)
            else:
                edge_index = np.stack([np.asarray(row), np.asarray(col)], axis=0)
            out_store.edge_index = edge_index
        elif store.is_edge_attr(key):
            if index is None:
                out_store[key] = None
                continue
            dim = 0
            if hasattr(store, '_parent') and store._parent() is not None:
                dim = store._parent().__cat_dim__(key, value, store)
            if perm is None:
                out_store[key] = index_select(value, index, dim=dim)
            else:
                sel_idx = perm[index] if hasattr(perm, '__getitem__') else index
                out_store[key] = index_select(value, sel_idx, dim=dim)


def filter_data(data: Data, node: Any, row: Any, col: Any, edge: Optional[Any] = None, perm: Optional[Any] = None) -> Data:
    out = copy.copy(data)
    out._store = copy.copy(data._store)
    filter_node_store_(data._store, out._store, node)
    filter_edge_store_(data._store, out._store, row, col, edge, perm)
    return out


def filter_hetero_data(
    data: HeteroData,
    node_dict: Dict[str, Any],
    row_dict: Dict[Tuple[str, str, str], Any],
    col_dict: Dict[Tuple[str, str, str], Any],
    edge_dict: Dict[Tuple[str, str, str], Optional[Any]],
    perm_dict: Optional[Dict[Tuple[str, str, str], Optional[Any]]] = None,
) -> HeteroData:
    out = copy.copy(data)
    out._node_store_dict = {k: copy.copy(v) for k, v in data._node_store_dict.items()}
    out._edge_store_dict = {k: copy.copy(v) for k, v in data._edge_store_dict.items()}

    for node_type in out.node_types:
        if node_type not in node_dict:
            node_dict[node_type] = torch.empty(0, dtype=torch.long) if torch is not None else np.empty(0, dtype=np.int64)
        filter_node_store_(data[node_type], out[node_type], node_dict[node_type])

    for edge_type in out.edge_types:
        canonical = data._to_canonical(*edge_type) if hasattr(data, '_to_canonical') else edge_type
        if canonical not in row_dict:
            empty_arr = torch.empty(0, dtype=torch.long) if torch is not None else np.empty(0, dtype=np.int64)
            row_dict[canonical] = empty_arr
            col_dict[canonical] = empty_arr
            edge_dict[canonical] = empty_arr

        filter_edge_store_(
            data[edge_type],
            out[edge_type],
            row_dict[canonical],
            col_dict[canonical],
            edge_dict[canonical],
            perm_dict.get(canonical, None) if perm_dict else None,
        )

    return out


def get_input_nodes(
    data: Union[Data, HeteroData],
    input_nodes: Any,
    input_id: Optional[Any] = None,
) -> Tuple[Optional[str], Any, Optional[Any]]:
    def to_index(nodes, in_id):
        if torch is not None and isinstance(nodes, torch.Tensor):
            if nodes.dtype == torch.bool:
                nodes = nodes.nonzero(as_tuple=False).view(-1)
                in_id = nodes if in_id is None else in_id
            return nodes, in_id
        if isinstance(nodes, np.ndarray) and nodes.dtype == bool:
            nodes = np.nonzero(nodes)[0]
            in_id = nodes if in_id is None else in_id
            return nodes, in_id
        if torch is not None and not isinstance(nodes, torch.Tensor):
            nodes = torch.tensor(nodes, dtype=torch.long)
        elif torch is None:
            nodes = np.asarray(nodes, dtype=np.int64)
        return nodes, in_id

    if isinstance(data, Data):
        if input_nodes is None:
            nodes = torch.arange(data.num_nodes) if torch is not None else np.arange(data.num_nodes)
            return None, nodes, None
        return None, *to_index(input_nodes, input_id)

    elif isinstance(data, HeteroData):
        assert input_nodes is not None
        if isinstance(input_nodes, str):
            num_nodes = data[input_nodes].num_nodes
            nodes = torch.arange(num_nodes) if torch is not None else np.arange(num_nodes)
            return input_nodes, nodes, None

        assert isinstance(input_nodes, (list, tuple)) and len(input_nodes) == 2
        node_type, input_nodes = input_nodes
        if input_nodes is None:
            num_nodes = data[node_type].num_nodes
            nodes = torch.arange(num_nodes) if torch is not None else np.arange(num_nodes)
            return node_type, nodes, None
        return node_type, *to_index(input_nodes, input_id)

    raise TypeError(f"Invalid data type: {type(data)}")


def get_edge_label_index(
    data: Union[Data, HeteroData],
    edge_label_index: Any,
) -> Tuple[Optional[Tuple[str, str, str]], Any]:
    if isinstance(data, Data):
        if edge_label_index is None:
            return None, data.edge_index
        return None, edge_label_index

    if isinstance(data, HeteroData):
        assert edge_label_index is not None
        if isinstance(edge_label_index, (list, tuple)) and len(edge_label_index) == 3 and isinstance(edge_label_index[0], str):
            edge_type = data._to_canonical(*edge_label_index)
            return edge_type, data[edge_type].edge_index

        assert isinstance(edge_label_index, (list, tuple)) and len(edge_label_index) == 2
        edge_type, edge_index = edge_label_index
        edge_type = data._to_canonical(*edge_type)
        if edge_index is None:
            return edge_type, data[edge_type].edge_index
        return edge_type, edge_index

    raise TypeError(f"Invalid data type: {type(data)}")


def infer_filter_per_worker(data: Any) -> bool:
    out = True
    if hasattr(data, 'is_cuda') and data.is_cuda:
        out = False
    return out
