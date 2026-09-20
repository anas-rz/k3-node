import copy
import weakref
from collections import defaultdict
from collections.abc import Mapping, MutableMapping, Sequence
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
from keras import ops

from k3_node.data.view import ItemsView, KeysView, ValuesView
from k3_node.utils.graph import coalesce, contains_isolated_nodes, has_self_loops, is_undirected

N_KEYS = {"x", "feat", "pos", "batch", "node_type", "n_id", "tf"}
E_KEYS = {"edge_index", "edge_weight", "edge_attr", "edge_type", "e_id"}


def is_tensor_like(x: Any) -> bool:
    if isinstance(x, np.ndarray):
        return True
    return hasattr(x, "shape") and hasattr(x, "dtype")


def to_numpy(x: Any) -> Any:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "numpy"):
        try:
            return x.numpy()
        except TypeError:
            if hasattr(x, "cpu"):
                return x.cpu().numpy()
            raise
    if hasattr(x, "cpu"):
        x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
    if hasattr(x, "_numpy"):
        return x._numpy()
    return np.asarray(x)


def get_shape(x: Any) -> Tuple[int, ...]:
    if hasattr(x, "shape"):
        return tuple(int(s) if s is not None else 0 for s in x.shape)
    if isinstance(x, (list, tuple)):
        return (len(x),)
    return ()


def recursive_apply(data: Any, func: Callable) -> Any:
    if is_tensor_like(data):
        return func(data)
    elif isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(*(recursive_apply(d, func) for d in data))
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return [recursive_apply(d, func) for d in data]
    elif isinstance(data, Mapping):
        return {key: recursive_apply(data[key], func) for key in data}
    else:
        try:
            return func(data)
        except Exception:
            return data


def recursive_apply_(data: Any, func: Callable) -> Any:
    if is_tensor_like(data):
        try:
            func(data)
        except Exception:
            pass
    elif isinstance(data, tuple) and hasattr(data, "_fields"):
        for value in data:
            recursive_apply_(value, func)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        for value in data:
            recursive_apply_(value, func)
    elif isinstance(data, Mapping):
        for value in data.values():
            recursive_apply_(value, func)
    else:
        try:
            func(data)
        except Exception:
            pass


class AttrType(Enum):
    NODE = "NODE"
    EDGE = "EDGE"
    OTHER = "OTHER"


class BaseStorage(MutableMapping):
    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        self._mapping: Dict[str, Any] = {}
        for key, value in (_mapping or {}).items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def _key(self) -> Any:
        return None

    def _pop_cache(self, key: str) -> None:
        for cache in getattr(self, "_cached_attr", {}).values():
            cache.discard(key)

    def __len__(self) -> int:
        return len(self._mapping)

    def __getattr__(self, key: str) -> Any:
        if key == "_mapping":
            self._mapping = {}
            return self._mapping
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'") from None

    def __setattr__(self, key: str, value: Any) -> None:
        propobj = getattr(self.__class__, key, None)
        if propobj is not None and getattr(propobj, "fset", None) is not None:
            propobj.fset(self, value)
        elif key == "_parent":
            self.__dict__[key] = weakref.ref(value) if value is not None else None
        elif key[:1] == "_":
            self.__dict__[key] = value
        else:
            self[key] = value

    def __delattr__(self, key: str) -> None:
        if key[:1] == "_":
            if key in self.__dict__:
                del self.__dict__[key]
        else:
            del self[key]

    def __getitem__(self, key: str) -> Any:
        return self._mapping[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._pop_cache(key)
        if value is None and key in self._mapping:
            del self._mapping[key]
        elif value is not None:
            self._mapping[key] = value

    def __delitem__(self, key: str) -> None:
        if key in self._mapping:
            self._pop_cache(key)
            del self._mapping[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._mapping)

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if key != "_cached_attr":
                out.__dict__[key] = value
        out._mapping = copy.copy(out._mapping)
        return out

    def __deepcopy__(self, memo=None):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            if key == "_parent":
                out.__dict__[key] = self.__dict__[key]
            elif key != "_cached_attr":
                out.__dict__[key] = copy.deepcopy(value, memo)
        out._mapping = copy.deepcopy(out._mapping, memo)
        return out

    def __getstate__(self) -> Dict[str, Any]:
        out = self.__dict__.copy()
        _parent = out.get("_parent", None)
        if _parent is not None:
            out["_parent"] = _parent()
        return out

    def __setstate__(self, mapping: Dict[str, Any]) -> None:
        for key, value in mapping.items():
            self.__dict__[key] = value
        _parent = self.__dict__.get("_parent", None)
        if _parent is not None:
            self.__dict__["_parent"] = weakref.ref(_parent)

    def __repr__(self) -> str:
        return repr(self._mapping)

    def _parent(self):
        parent_ref = self.__dict__.get("_parent", None)
        return parent_ref() if parent_ref is not None else None

    def keys(self, *args: str) -> KeysView:
        return KeysView(self._mapping, *args)

    def values(self, *args: str) -> ValuesView:
        return ValuesView(self._mapping, *args)

    def items(self, *args: str) -> ItemsView:
        return ItemsView(self._mapping, *args)

    def to_dict(self) -> Dict[str, Any]:
        return copy.copy(self._mapping)

    def apply_(self, func: Callable, *args: str):
        for value in self.values(*args):
            recursive_apply_(value, func)
        return self

    def apply(self, func: Callable, *args: str):
        for key, value in self.items(*args):
            self[key] = recursive_apply(value, func)
        return self

    def to(self, *args, **kwargs):
        def _to(x):
            if hasattr(x, "to"):
                return x.to(*args, **kwargs)
            return x

        return self.apply(_to)

    def to_backend(self, backend: Optional[str] = None):
        for key in list(self.keys()):
            val = self[key]
            if is_tensor_like(val):
                v_np = to_numpy(val)
                if v_np.dtype == np.bool_:
                    self[key] = ops.convert_to_tensor(v_np, dtype="bool")
                elif np.issubdtype(v_np.dtype, np.integer):
                    self[key] = ops.convert_to_tensor(v_np, dtype="int64")
                elif np.issubdtype(v_np.dtype, np.floating):
                    self[key] = ops.convert_to_tensor(v_np, dtype="float32")
                else:
                    self[key] = ops.convert_to_tensor(v_np)
        return self

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    def requires_grad_(self, *args: str):
        def _req(x):
            if hasattr(x, "requires_grad_"):
                x.requires_grad_()
            elif hasattr(x, "requires_grad"):
                x.requires_grad = True

        return self.apply_(_req, *args)

    def contiguous(self, *args: str):
        def _cont(x):
            if hasattr(x, "contiguous"):
                return x.contiguous()
            return x

        return self.apply(_cont, *args)


class NodeStorage(BaseStorage):
    @property
    def _key(self) -> Any:
        return self.__dict__.get("_key", None)

    @property
    def num_nodes(self) -> int:
        if "num_nodes" in self:
            return int(self["num_nodes"])
        parent = self._parent()
        for key, value in self.items():
            if is_tensor_like(value) and key in N_KEYS:
                cat_dim = parent.__cat_dim__(key, value, self) if parent is not None else 0
                return get_shape(value)[cat_dim]
        for key, value in self.items():
            if is_tensor_like(value) and "node" in key:
                cat_dim = parent.__cat_dim__(key, value, self) if parent is not None else 0
                return get_shape(value)[cat_dim]
        return 0

    @property
    def num_node_features(self) -> int:
        x = self.get("x")
        if x is not None and is_tensor_like(x):
            shape = get_shape(x)
            return 1 if len(shape) == 1 else shape[-1]
        return 0

    @property
    def num_features(self) -> int:
        return self.num_node_features

    def is_node_attr(self, key: str) -> bool:
        if "_cached_attr" not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.NODE]:
            return True
        if key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self.get(key)
        if value is None:
            return False

        if isinstance(value, (list, tuple)) and len(value) == self.num_nodes:
            self._cached_attr[AttrType.NODE].add(key)
            return True

        if not is_tensor_like(value):
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        shape = get_shape(value)
        if len(shape) == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        parent = self._parent()
        cat_dim = parent.__cat_dim__(key, value, self) if parent is not None else 0
        if shape[cat_dim] != self.num_nodes:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        self._cached_attr[AttrType.NODE].add(key)
        return True

    def is_edge_attr(self, key: str) -> bool:
        return False

    def node_attrs(self) -> List[str]:
        return [key for key in self.keys() if self.is_node_attr(key)]


class EdgeStorage(BaseStorage):
    @property
    def _key(self) -> Any:
        return self.__dict__.get("_key", None)

    @property
    def edge_index(self):
        if "edge_index" in self:
            return self["edge_index"]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute 'edge_index'")

    @edge_index.setter
    def edge_index(self, edge_index) -> None:
        self["edge_index"] = edge_index

    @property
    def num_edges(self) -> int:
        if "num_edges" in self:
            return int(self["num_edges"])
        parent = self._parent()
        for key, value in self.items():
            if is_tensor_like(value) and key in E_KEYS:
                cat_dim = parent.__cat_dim__(key, value, self) if parent is not None else -1
                return get_shape(value)[cat_dim]
        for key, value in self.items():
            if is_tensor_like(value) and "edge" in key:
                cat_dim = parent.__cat_dim__(key, value, self) if parent is not None else -1
                return get_shape(value)[cat_dim]
        return 0

    @property
    def num_edge_features(self) -> int:
        edge_attr = self.get("edge_attr")
        if edge_attr is not None and is_tensor_like(edge_attr):
            shape = get_shape(edge_attr)
            return 1 if len(shape) == 1 else shape[-1]
        return 0

    @property
    def num_features(self) -> int:
        return self.num_edge_features

    def size(self, dim: Optional[int] = None) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        parent = self._parent()
        if self._key is None or parent is None:
            num = self.num_edges
            res = (num, num)
            return res if dim is None else res[dim]
        size = (parent[self._key[0]].num_nodes, parent[self._key[-1]].num_nodes)
        return size if dim is None else size[dim]

    def is_node_attr(self, key: str) -> bool:
        return False

    def is_edge_attr(self, key: str) -> bool:
        if "_cached_attr" not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.EDGE]:
            return True
        if key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self.get(key)
        if value is None:
            return False

        if isinstance(value, (list, tuple)) and len(value) == self.num_edges:
            self._cached_attr[AttrType.EDGE].add(key)
            return True

        if not is_tensor_like(value):
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        shape = get_shape(value)
        if len(shape) == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        parent = self._parent()
        cat_dim = parent.__cat_dim__(key, value, self) if parent is not None else -1
        if shape[cat_dim] != self.num_edges:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        self._cached_attr[AttrType.EDGE].add(key)
        return True

    def edge_attrs(self) -> List[str]:
        return [key for key in self.keys() if self.is_edge_attr(key)]

    def is_coalesced(self) -> bool:
        if "edge_index" in self:
            edge_index = self.edge_index
            new_edge_index, _ = coalesce(edge_index)
            orig_np = ops.convert_to_numpy(edge_index)
            new_np = ops.convert_to_numpy(new_edge_index)
            return orig_np.shape == new_np.shape and np.array_equal(orig_np, new_np)
        return True

    def coalesce(self, reduce: str = "add"):
        if "edge_index" in self:
            self.edge_index, self.edge_attr = coalesce(
                self.edge_index,
                edge_attr=self.get("edge_attr"),
                reduce=reduce,
            )
        return self

    def has_self_loops(self) -> bool:
        if self.is_bipartite() or "edge_index" not in self:
            return False
        return has_self_loops(self.edge_index)

    def has_isolated_nodes(self) -> bool:
        if "edge_index" not in self:
            return False
        parent = self._parent()
        num_nodes = parent[self._key[-1]].num_nodes if parent and self._key else None
        return contains_isolated_nodes(self.edge_index, num_nodes=num_nodes)

    def is_undirected(self) -> bool:
        if self.is_bipartite() or "edge_index" not in self:
            return False
        return is_undirected(self.edge_index, edge_attr=self.get("edge_attr"))

    def is_directed(self) -> bool:
        return not self.is_undirected()

    def is_bipartite(self) -> bool:
        return self._key is not None and isinstance(self._key, tuple) and self._key[0] != self._key[-1]


class GlobalStorage(NodeStorage, EdgeStorage):
    @property
    def _key(self) -> Any:
        return None

    @property
    def num_features(self) -> int:
        return self.num_node_features

    def size(self, dim: Optional[int] = None) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        size = (self.num_nodes, self.num_nodes)
        return size if dim is None else size[dim]

    def is_node_attr(self, key: str) -> bool:
        if "_cached_attr" not in self.__dict__:
            self._cached_attr: Dict[AttrType, Set[str]] = defaultdict(set)

        if key in self._cached_attr[AttrType.NODE]:
            return True
        if key in self._cached_attr[AttrType.EDGE] or key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self.get(key)
        if value is None:
            return False

        if isinstance(value, (list, tuple)) and len(value) == self.num_nodes:
            self._cached_attr[AttrType.NODE].add(key)
            return True

        if not is_tensor_like(value):
            return False

        shape = get_shape(value)
        if len(shape) == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        parent = self._parent()
        cat_dim = parent.__cat_dim__(key, value, self) if parent is not None else 0
        if not isinstance(cat_dim, int):
            return False

        num_nodes, num_edges = self.num_nodes, self.num_edges

        if shape[cat_dim] != num_nodes:
            if shape[cat_dim] == num_edges:
                self._cached_attr[AttrType.EDGE].add(key)
            else:
                self._cached_attr[AttrType.OTHER].add(key)
            return False

        if num_nodes != num_edges:
            self._cached_attr[AttrType.NODE].add(key)
            return True

        if "edge" not in key:
            self._cached_attr[AttrType.NODE].add(key)
            return True
        else:
            self._cached_attr[AttrType.EDGE].add(key)
            return False

    def is_edge_attr(self, key: str) -> bool:
        if "_cached_attr" not in self.__dict__:
            self._cached_attr = defaultdict(set)

        if key in self._cached_attr[AttrType.EDGE]:
            return True
        if key in self._cached_attr[AttrType.NODE] or key in self._cached_attr[AttrType.OTHER]:
            return False

        value = self.get(key)
        if value is None:
            return False

        if isinstance(value, (list, tuple)) and len(value) == self.num_edges:
            self._cached_attr[AttrType.EDGE].add(key)
            return True

        if not is_tensor_like(value):
            return False

        shape = get_shape(value)
        if len(shape) == 0:
            self._cached_attr[AttrType.OTHER].add(key)
            return False

        parent = self._parent()
        cat_dim = parent.__cat_dim__(key, value, self) if parent is not None else -1
        if not isinstance(cat_dim, int):
            return False

        num_nodes, num_edges = self.num_nodes, self.num_edges

        if shape[cat_dim] != num_edges:
            if shape[cat_dim] == num_nodes:
                self._cached_attr[AttrType.NODE].add(key)
            else:
                self._cached_attr[AttrType.OTHER].add(key)
            return False

        if num_edges != num_nodes:
            self._cached_attr[AttrType.EDGE].add(key)
            return True

        if "edge" in key:
            self._cached_attr[AttrType.EDGE].add(key)
            return True
        else:
            self._cached_attr[AttrType.NODE].add(key)
            return False
