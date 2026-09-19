import numpy as np
from keras import ops
from typing import Any, List, Optional, Tuple, Union

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def is_torch_tensor(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def to_numpy(x: Any) -> np.ndarray:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if is_torch_tensor(x):
        return x.detach().cpu().numpy()
    return ops.convert_to_numpy(x)


def as_tensor(x: Any, dtype: Optional[Any] = None, like: Optional[Any] = None) -> Any:
    if x is None:
        return None
    if like is not None and is_torch_tensor(like):
        if isinstance(dtype, str):
            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "float16": torch.float16,
                "int64": torch.int64,
                "int32": torch.int32,
                "int16": torch.int16,
                "int8": torch.int8,
                "uint8": torch.uint8,
                "bool": torch.bool,
            }
            target_dtype = dtype_map.get(dtype, getattr(torch, dtype, None))
        else:
            target_dtype = dtype or like.dtype
        target_device = like.device
        if is_torch_tensor(x):
            return x.to(device=target_device, dtype=target_dtype)
        np_arr = to_numpy(x)
        return torch.as_tensor(np_arr, dtype=target_dtype, device=target_device)
    if is_torch_tensor(x):
        if dtype is not None:
            if isinstance(dtype, str):
                target_dtype = getattr(torch, dtype, None) or getattr(torch, dtype.replace("torch.", ""), None)
            else:
                target_dtype = dtype
            return x.to(dtype=target_dtype)
        return x
    if isinstance(x, np.ndarray) and like is not None and isinstance(like, np.ndarray):
        return np.asarray(x, dtype=dtype or like.dtype)
    return ops.convert_to_tensor(x, dtype=dtype)


def match_tensor(new_val: Any, reference: Any, dtype: Optional[Any] = None) -> Any:
    if reference is None:
        if is_torch_tensor(new_val):
            return new_val
        return ops.convert_to_tensor(new_val, dtype=dtype)
    return as_tensor(new_val, dtype=dtype, like=reference)


def to_undirected(
    edge_index: Any,
    edge_attr: Optional[Any] = None,
    num_nodes: Optional[int] = None,
    reduce: str = "add",
) -> Union[Any, Tuple[Any, Any]]:
    from k3_node.utils.graph import coalesce

    ei_np = to_numpy(edge_index)
    if ei_np.shape[1] == 0:
        if edge_attr is None:
            return edge_index
        return edge_index, edge_attr

    row, col = ei_np[0], ei_np[1]
    rev_row = np.concatenate([row, col], axis=0)
    rev_col = np.concatenate([col, row], axis=0)
    rev_edge_index = np.stack([rev_row, rev_col], axis=0)

    if edge_attr is not None:
        attr_np = to_numpy(edge_attr)
        if attr_np.ndim == 1:
            rev_attr = np.concatenate([attr_np, attr_np], axis=0)
        else:
            rev_attr = np.concatenate([attr_np, attr_np], axis=0)
        out_ei, out_attr = coalesce(rev_edge_index, rev_attr, num_nodes=num_nodes, reduce=reduce)
        return match_tensor(out_ei, edge_index), match_tensor(out_attr, edge_attr)
    else:
        out_ei, _ = coalesce(rev_edge_index, None, num_nodes=num_nodes, reduce=reduce)
        return match_tensor(out_ei, edge_index)
