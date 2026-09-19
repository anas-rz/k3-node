from typing import Any, Type, TypeVar

from keras import ops

from k3_node.data.data import BaseData, Data
from k3_node.data.hetero_data import HeteroData
from k3_node.data.storage import BaseStorage, get_shape, is_tensor_like

T = TypeVar("T")


def narrow(tensor, dim: int, start: int, length: int):
    shape = list(get_shape(tensor))
    if dim < 0:
        dim = len(shape) + dim
    slices = [slice(None)] * len(shape)
    slices[dim] = slice(start, start + length)
    return tensor[tuple(slices)]


def separate(
    cls: Type[T],
    batch: Any,
    idx: int,
    slice_dict: Any,
    inc_dict: Any = None,
    decrement: bool = True,
) -> T:
    is_hetero = isinstance(batch, HeteroData)
    data = HeteroData() if is_hetero else Data()

    if not is_hetero:
        batch_store = batch._store
        data_store = data._store

        for attr in slice_dict.keys():
            if attr not in batch_store:
                continue
            slices = slice_dict[attr]
            incs = inc_dict[attr] if (decrement and inc_dict is not None and attr in inc_dict) else None

            val = batch_store[attr]
            if is_tensor_like(val):
                start = int(slices[idx])
                end = int(slices[idx + 1])
                length = end - start
                cat_dim = batch.__cat_dim__(attr, val, batch_store)
                sub_val = narrow(val, cat_dim, start, length)
                if decrement and incs is not None and idx < len(incs):
                    inc_val = incs[idx]
                    if inc_val != 0:
                        inc_t = ops.convert_to_tensor(inc_val, dtype=sub_val.dtype)
                        sub_val = sub_val - inc_t
                data_store[attr] = sub_val
            elif isinstance(val, (list, tuple)) and len(val) > idx:
                data_store[attr] = val[idx]
            else:
                data_store[attr] = val

        if hasattr(batch_store, "_num_nodes") and idx < len(batch_store._num_nodes):
            data_store.num_nodes = batch_store._num_nodes[idx]

    else:
        # Heterogeneous separate
        for node_type in batch.node_types:
            batch_store = batch[node_type]
            data_store = data[node_type]
            store_slice_dict = slice_dict.get(node_type, {})
            store_inc_dict = inc_dict.get(node_type, {}) if decrement and inc_dict else {}

            for attr in store_slice_dict.keys():
                if attr not in batch_store:
                    continue
                slices = store_slice_dict[attr]
                val = batch_store[attr]
                if is_tensor_like(val):
                    start = int(slices[idx])
                    end = int(slices[idx + 1])
                    length = end - start
                    cat_dim = batch.__cat_dim__(attr, val, batch_store)
                    sub_val = narrow(val, cat_dim, start, length)
                    data_store[attr] = sub_val
                elif isinstance(val, (list, tuple)) and len(val) > idx:
                    data_store[attr] = val[idx]

        for edge_type in batch.edge_types:
            batch_store = batch[edge_type]
            data_store = data[edge_type]
            store_slice_dict = slice_dict.get(edge_type, {})
            store_inc_dict = inc_dict.get(edge_type, {}) if decrement and inc_dict else {}

            for attr in store_slice_dict.keys():
                if attr not in batch_store:
                    continue
                slices = store_slice_dict[attr]
                incs = store_inc_dict.get(attr) if decrement else None
                val = batch_store[attr]
                if is_tensor_like(val):
                    start = int(slices[idx])
                    end = int(slices[idx + 1])
                    length = end - start
                    cat_dim = batch.__cat_dim__(attr, val, batch_store)
                    sub_val = narrow(val, cat_dim, start, length)
                    if decrement and incs is not None and idx < len(incs):
                        inc_val = incs[idx]
                        if is_tensor_like(inc_val) or (isinstance(inc_val, np.ndarray) and np.any(inc_val != 0)):
                            if hasattr(inc_val, "ndim") and inc_val.ndim == 1:
                                inc_val = inc_val[:, None]
                            inc_t = ops.convert_to_tensor(inc_val, dtype=sub_val.dtype)
                            sub_val = sub_val - inc_t
                    data_store[attr] = sub_val
                elif isinstance(val, (list, tuple)) and len(val) > idx:
                    data_store[attr] = val[idx]

    return data
