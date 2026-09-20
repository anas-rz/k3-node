from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
from keras import ops

from k3_node.data.data import BaseData, Data
from k3_node.data.hetero_data import HeteroData
from k3_node.data.storage import BaseStorage, NodeStorage, get_shape, is_tensor_like

T = TypeVar("T")
SliceDictType = Dict[str, Any]
IncDictType = Dict[str, Any]


try:
    import torch
except ImportError:
    torch = None


def _concat_tensors(values: List[Any], axis: int) -> Any:
    if torch is not None and isinstance(values[0], torch.Tensor):
        return torch.cat(values, dim=axis)
    elif isinstance(values[0], np.ndarray):
        return np.concatenate(values, axis=axis)
    return ops.concatenate(values, axis=axis)


def _batch_and_ptr(slices: Sequence, num_graphs: int, is_torch: bool = False) -> Tuple[Any, Any]:
    repeats = [int(slices[i + 1] - slices[i]) for i in range(num_graphs)]
    batch = np.repeat(np.arange(num_graphs), repeats)
    ptr = np.array(slices)
    if is_torch and torch is not None:
        return torch.from_numpy(batch).to(dtype=torch.long), torch.from_numpy(ptr).to(dtype=torch.long)
    return ops.convert_to_tensor(batch, dtype="int64"), ops.convert_to_tensor(ptr, dtype="int64")


def collate(
    cls: Type[T],
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[Iterable[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
) -> Tuple[T, SliceDictType, IncDictType]:
    if not isinstance(data_list, (list, tuple)):
        data_list = list(data_list)

    if len(data_list) == 0:
        raise ValueError("Cannot collate an empty list of data objects")

    is_hetero = isinstance(data_list[0], HeteroData)
    out = cls(_base_cls=data_list[0].__class__) if hasattr(cls, "_base_cls") else cls()

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    slice_dict: SliceDictType = {}
    inc_dict: IncDictType = {}

    if not is_hetero:
        # Homogeneous Data
        out_store = out._store
        all_keys = set()
        for d in data_list:
            all_keys.update(d.keys())

        # Determine num_nodes for each graph in batch
        num_nodes_list = [d.num_nodes or 0 for d in data_list]
        out_store.num_nodes = sum(num_nodes_list)

        for key in all_keys:
            if key in exclude_keys or key == "num_nodes" or key == "ptr":
                continue

            values = [d.get(key) for d in data_list if key in d]
            if len(values) != len(data_list):
                continue  # Key not present in all graphs

            elem = values[0]
            if is_tensor_like(elem):
                cat_dim = data_list[0].__cat_dim__(key, elem, data_list[0]._store)
                elem_shape = get_shape(elem)
                if len(elem_shape) == 0 or cat_dim is None:
                    cat_dim = 0
                    values = [ops.expand_dims(v, axis=0) for v in values]
                elif cat_dim < 0:
                    cat_dim = len(elem_shape) + cat_dim
                target_rank = len(elem_shape)
                normalized_values = []
                for v in values:
                    v_shape = get_shape(v)
                    if len(v_shape) != target_rank:
                        if key in ("edge_index", "adj_t") and (len(v_shape) == 0 or v_shape == (0,)):
                            v = ops.zeros((2, 0), dtype="int64")
                        elif len(v_shape) == 1 and v_shape[0] == 0:
                            v = ops.zeros((0,) + elem_shape[1:], dtype="float32")
                    normalized_values.append(v)
                values = normalized_values

                sizes = [get_shape(v)[cat_dim] for v in values]
                slices = np.cumsum([0] + sizes)

                incs = []
                cum_inc = 0
                for i in range(len(values)):
                    incs.append(cum_inc)
                    inc_val = data_list[i].__inc__(key, values[i], data_list[i]._store)
                    cum_inc += inc_val

                if increment and cum_inc != 0:
                    offset_values = []
                    for val, inc in zip(values, incs):
                        if inc != 0:
                            if torch is not None and isinstance(val, torch.Tensor):
                                inc_t = torch.as_tensor(inc, dtype=val.dtype, device=val.device)
                                offset_values.append(val + inc_t)
                            else:
                                inc_t = ops.convert_to_tensor(inc, dtype=val.dtype)
                                offset_values.append(val + inc_t)
                        else:
                            offset_values.append(val)
                    values = offset_values

                out_val = _concat_tensors(values, axis=cat_dim)
                out_store[key] = out_val
                slice_dict[key] = slices
                inc_dict[key] = np.array(incs)

                if key in follow_batch:
                    is_t = torch is not None and isinstance(values[0], torch.Tensor)
                    batch_vec, ptr_vec = _batch_and_ptr(slices, len(data_list), is_torch=is_t)
                    out_store[f"{key}_batch"] = batch_vec
                    out_store[f"{key}_ptr"] = ptr_vec
            else:
                out_store[key] = values
                slice_dict[key] = np.arange(len(values) + 1)
                inc_dict[key] = np.zeros(len(values))

        if add_batch:
            repeats = num_nodes_list
            batch_arr = np.repeat(np.arange(len(repeats)), repeats)
            ptr_arr = np.cumsum([0] + repeats)
            is_t = torch is not None and any(isinstance(getattr(d, 'x', None), torch.Tensor) or isinstance(getattr(d, 'edge_index', None), torch.Tensor) for d in data_list)
            if is_t and torch is not None:
                out_store.batch = torch.from_numpy(batch_arr).to(dtype=torch.long)
                out_store.ptr = torch.from_numpy(ptr_arr).to(dtype=torch.long)
            else:
                out_store.batch = ops.convert_to_tensor(batch_arr, dtype="int64")
                out_store.ptr = ops.convert_to_tensor(ptr_arr, dtype="int64")

    else:
        # Heterogeneous Data
        node_types = data_list[0].node_types
        edge_types = data_list[0].edge_types

        # Collate each node type
        for n_type in node_types:
            out_store = out[n_type]
            all_keys = set()
            for d in data_list:
                all_keys.update(d[n_type].keys())

            num_nodes_list = [d[n_type].num_nodes or 0 for d in data_list]
            out_store.num_nodes = sum(num_nodes_list)

            store_slice_dict = {}
            store_inc_dict = {}

            for key in all_keys:
                if key in exclude_keys or key == "num_nodes" or key == "ptr":
                    continue
                values = [d[n_type].get(key) for d in data_list if key in d[n_type]]
                if len(values) != len(data_list):
                    continue

                elem = values[0]
                if is_tensor_like(elem):
                    cat_dim = data_list[0].__cat_dim__(key, elem, data_list[0][n_type])
                    elem_shape = get_shape(elem)
                    if len(elem_shape) == 0 or cat_dim is None:
                        cat_dim = 0
                        values = [ops.expand_dims(v, axis=0) for v in values]
                    elif cat_dim < 0:
                        cat_dim = len(elem_shape) + cat_dim

                    sizes = [get_shape(v)[cat_dim] for v in values]
                    slices = np.cumsum([0] + sizes)
                    incs = [0] * len(values)

                    out_val = _concat_tensors(values, axis=cat_dim)
                    out_store[key] = out_val
                    store_slice_dict[key] = slices
                    store_inc_dict[key] = np.array(incs)

                    if key in follow_batch:
                        is_t = torch is not None and isinstance(values[0], torch.Tensor)
                        batch_vec, ptr_vec = _batch_and_ptr(slices, len(data_list), is_torch=is_t)
                        out_store[f"{key}_batch"] = batch_vec
                        out_store[f"{key}_ptr"] = ptr_vec
                else:
                    out_store[key] = values

            if add_batch:
                repeats = num_nodes_list
                batch_arr = np.repeat(np.arange(len(repeats)), repeats)
                ptr_arr = np.cumsum([0] + repeats)
                is_t = torch is not None and any(isinstance(getattr(d[n_type], 'x', None), torch.Tensor) for d in data_list)
                if is_t and torch is not None:
                    out_store.batch = torch.from_numpy(batch_arr).to(dtype=torch.long)
                    out_store.ptr = torch.from_numpy(ptr_arr).to(dtype=torch.long)
                else:
                    out_store.batch = ops.convert_to_tensor(batch_arr, dtype="int64")
                    out_store.ptr = ops.convert_to_tensor(ptr_arr, dtype="int64")

            slice_dict[n_type] = store_slice_dict
            inc_dict[n_type] = store_inc_dict

        # Collate each edge type
        for e_type in edge_types:
            out_store = out[e_type]
            all_keys = set()
            for d in data_list:
                all_keys.update(d[e_type].keys())

            store_slice_dict = {}
            store_inc_dict = {}

            src_type, _, dst_type = e_type

            for key in all_keys:
                if key in exclude_keys:
                    continue
                values = [d[e_type].get(key) for d in data_list if key in d[e_type]]
                if len(values) != len(data_list):
                    continue

                elem = values[0]
                if is_tensor_like(elem):
                    cat_dim = data_list[0].__cat_dim__(key, elem, data_list[0][e_type])
                    elem_shape = get_shape(elem)
                    if len(elem_shape) == 0 or cat_dim is None:
                        cat_dim = 0
                        values = [ops.expand_dims(v, axis=0) for v in values]
                    elif cat_dim < 0:
                        cat_dim = len(elem_shape) + cat_dim

                    sizes = [get_shape(v)[cat_dim] for v in values]
                    slices = np.cumsum([0] + sizes)

                    if key == "edge_index" and increment:
                        src_incs = np.cumsum([0] + [d[src_type].num_nodes or 0 for d in data_list[:-1]])
                        dst_incs = np.cumsum([0] + [d[dst_type].num_nodes or 0 for d in data_list[:-1]])
                        offset_values = []
                        for i, val in enumerate(values):
                            inc_arr = np.array([[src_incs[i]], [dst_incs[i]]], dtype=np.int64)
                            if torch is not None and isinstance(val, torch.Tensor):
                                inc_t = torch.as_tensor(inc_arr, dtype=val.dtype, device=val.device)
                            else:
                                inc_t = ops.convert_to_tensor(inc_arr, dtype=val.dtype)
                            offset_values.append(val + inc_t)
                        values = offset_values
                        incs = np.stack([src_incs, dst_incs], axis=1)
                    else:
                        incs = np.zeros(len(values))

                    out_val = _concat_tensors(values, axis=cat_dim)
                    out_store[key] = out_val
                    store_slice_dict[key] = slices
                    store_inc_dict[key] = incs
                else:
                    out_store[key] = values

            slice_dict[e_type] = store_slice_dict
            inc_dict[e_type] = store_inc_dict

    return out, slice_dict, inc_dict
