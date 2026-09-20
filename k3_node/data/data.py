import collections
import copy
import warnings
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Any, Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from keras import ops

from k3_node.data.storage import (
    BaseStorage,
    EdgeStorage,
    GlobalStorage,
    NodeStorage,
    get_shape,
    is_tensor_like,
    recursive_apply,
    recursive_apply_,
)
from k3_node.utils.graph import coalesce, contains_isolated_nodes, has_self_loops, is_undirected, subgraph


def size_repr(key: Any, value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if is_tensor_like(value):
        shape = get_shape(value)
        if len(shape) == 0:
            out = str(ops.convert_to_numpy(value).item())
        else:
            out = str(list(shape))
    elif isinstance(value, str):
        out = f"'{value}'"
    elif isinstance(value, (Sequence, set)) and not isinstance(value, str):
        out = str([len(value)])
    elif isinstance(value, Mapping) and len(value) == 0:
        out = "{}"
    elif isinstance(value, Mapping) and len(value) == 1 and not isinstance(list(value.values())[0], Mapping):
        lines = [size_repr(k, v, 0) for k, v in value.items()]
        out = "{ " + ", ".join(lines) + " }"
    elif isinstance(value, Mapping):
        lines = [size_repr(k, v, indent + 2) for k, v in value.items()]
        out = "{\n" + ",\n".join(lines) + ",\n" + pad + "}"
    else:
        out = str(value)

    key = str(key).replace("'", "")
    return f"{pad}{key}={out}"


class BaseData:
    def __getattr__(self, key: str) -> Any:
        raise NotImplementedError

    def __setattr__(self, key: str, value: Any):
        raise NotImplementedError

    def __delattr__(self, key: str):
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        raise NotImplementedError

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError

    def __delitem__(self, key: str):
        raise NotImplementedError

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memo=None):
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    @property
    def stores(self) -> List[BaseStorage]:
        raise NotImplementedError

    @property
    def node_stores(self) -> List[NodeStorage]:
        raise NotImplementedError

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    def to_namedtuple(self) -> NamedTuple:
        raise NotImplementedError

    def to_backend(self, backend: Optional[str] = None) -> "BaseData":
        for store in self.stores:
            store.to_backend(backend)
        return self

    def update(self, data: "BaseData") -> "BaseData":
        for store, other_store in zip(self.stores, data.stores):
            for key, value in other_store.items():
                store[key] = value
        return self

    def __len__(self) -> int:
        return len(self.keys())

    def __contains__(self, key: str) -> bool:
        return key in self.keys()

    def keys(self, *args: str) -> List[str]:
        out = []
        for store in self.stores:
            out.extend(list(store.keys(*args)))
        return list(set(out))

    def values(self, *args: str) -> List[Any]:
        return [self[k] for k in self.keys(*args)]

    def items(self, *args: str) -> List[Tuple[str, Any]]:
        return [(k, self[k]) for k in self.keys(*args)]

    @property
    def num_nodes(self) -> Optional[int]:
        try:
            return sum([v.num_nodes for v in self.node_stores])
        except TypeError:
            return None

    @property
    def num_edges(self) -> int:
        return sum([v.num_edges for v in self.edge_stores])

    def node_attrs(self) -> List[str]:
        return list(set(chain(*[s.node_attrs() for s in self.node_stores])))

    def edge_attrs(self) -> List[str]:
        return list(set(chain(*[s.edge_attrs() for s in self.edge_stores])))


class Data(BaseData):
    """A data object describing a homogeneous graph."""

    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        pos=None,
        **kwargs,
    ):
        self.__dict__["_store"] = GlobalStorage(_parent=self)
        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y
        if pos is not None:
            self.pos = pos
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getattr__(self, key: str) -> Any:
        if "_store" not in self.__dict__:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        try:
            return getattr(self._store, key)
        except AttributeError:
            if key in ('x', 'edge_index', 'edge_attr', 'edge_weight', 'y', 'pos', 'face', 'normal', 'batch'):
                return None
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'") from None

    def __setattr__(self, key: str, value: Any):
        if key == "_store":
            self.__dict__["_store"] = value
        elif "_store" in self.__dict__:
            setattr(self._store, key, value)
        else:
            self.__dict__[key] = value

    def __delattr__(self, key: str):
        if key == "_store":
            del self.__dict__["_store"]
        elif "_store" in self.__dict__:
            delattr(self._store, key)
        else:
            del self.__dict__[key]

    def __getitem__(self, key: str) -> Any:
        return self._store[key]

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value

    def __delitem__(self, key: str):
        del self._store[key]

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            out.__dict__[k] = v
        out._store = copy.copy(self._store)
        out._store._parent = weakref_self = None
        out.__dict__["_store"].__dict__["_parent"] = weakref_self
        setattr(out._store, "_parent", out)
        return out

    def __deepcopy__(self, memo=None):
        out = self.__class__.__new__(self.__class__)
        for k, v in self.__dict__.items():
            if k == "_store":
                out.__dict__[k] = copy.deepcopy(v, memo)
            else:
                out.__dict__[k] = copy.deepcopy(v, memo)
        setattr(out._store, "_parent", out)
        return out

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, mapping: Dict[str, Any]):
        import weakref

        for key, value in mapping.items():
            self.__dict__[key] = value
        if "_store" in self.__dict__ and self._store is not None:
            self._store.__dict__["_parent"] = weakref.ref(self)

    def clone(self) -> "Data":
        return copy.deepcopy(self)

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def __call__(self, *args: str) -> Iterator[Tuple[str, Any]]:
        yield from self._store.items(*args)

    @property
    def num_features(self) -> int:
        return self._store.num_features

    @property
    def num_node_features(self) -> int:
        return self._store.num_node_features

    @property
    def num_edge_features(self) -> int:
        return self._store.num_edge_features

    @property
    def num_node_types(self) -> int:
        return 1

    @property
    def num_edge_types(self) -> int:
        return 1

    @property
    def num_classes(self) -> Optional[int]:
        y = self.get("y")
        if y is not None and is_tensor_like(y):
            y_np = ops.convert_to_numpy(y)
            if np.issubdtype(y_np.dtype, np.integer):
                return int(np.max(y_np)) + 1
        return None

    def is_directed(self) -> bool:
        return self._store.is_directed()

    def is_undirected(self) -> bool:
        return self._store.is_undirected()

    def has_self_loops(self) -> bool:
        return self._store.has_self_loops()

    def has_isolated_nodes(self) -> bool:
        return self._store.has_isolated_nodes()

    def is_coalesced(self) -> bool:
        return self._store.is_coalesced()

    def coalesce(self, reduce: str = "add"):
        self._store.coalesce(reduce=reduce)
        return self

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "batch" in key:
            return int(value.max()) + 1 if is_tensor_like(value) and value.ndim > 0 and value.shape[0] > 0 else 0
        if "index" in key or "face" in key:
            return self.num_nodes or 0
        return 0

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> int:
        if key in ("edge_index", "adj_t"):
            return -1
        if is_tensor_like(value) and len(get_shape(value)) == 2 and get_shape(value)[0] == 2 and "index" in key:
            return -1
        return 0

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        fields = sorted(list(self.keys()))
        DataTuple = collections.namedtuple("DataTuple", fields)
        return DataTuple(**{f: self[f] for f in fields})

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]) -> "Data":
        return cls(**mapping)

    def apply(self, func: Callable, *keys: str) -> "Data":
        self._store.apply(func, *keys)
        return self

    def apply_(self, func: Callable, *keys: str) -> "Data":
        self._store.apply_(func, *keys)
        return self

    def to(self, *args, **kwargs) -> "Data":
        self._store.to(*args, **kwargs)
        return self

    def to_backend(self, backend: Optional[str] = None) -> "Data":
        self._store.to_backend(backend)
        return self

    def cpu(self) -> "Data":
        self._store.cpu()
        return self

    def cuda(self) -> "Data":
        self._store.cuda()
        return self

    def requires_grad_(self, *keys: str) -> "Data":
        self._store.requires_grad_(*keys)
        return self

    def contiguous(self, *keys: str) -> "Data":
        self._store.contiguous(*keys)
        return self

    def subgraph(self, subset) -> "Data":
        """Returns the induced subgraph for subset nodes."""
        data = copy.copy(self)
        num_nodes = self.num_nodes
        sub_edge_index, sub_edge_attr = subgraph(
            subset,
            self.edge_index,
            edge_attr=self.get("edge_attr"),
            relabel_nodes=True,
            num_nodes=num_nodes,
        )
        data.edge_index = sub_edge_index
        if sub_edge_attr is not None:
            data.edge_attr = sub_edge_attr
        subset_np = ops.convert_to_numpy(subset)
        if subset_np.dtype == bool:
            indices = np.where(subset_np)[0]
        else:
            indices = subset_np

        for key in self.node_attrs():
            val = self[key]
            if is_tensor_like(val):
                data[key] = ops.take(val, indices, axis=self.__cat_dim__(key, val))
        return data

    def to_heterogeneous(self, node_type: str = "0", edge_type: Tuple[str, str, str] = ("0", "0", "0")):
        from k3_node.data.hetero_data import HeteroData

        hetero = HeteroData()
        if hasattr(self, "edge_type") and self.edge_type is not None:
            edge_type_np = ops.convert_to_numpy(self.edge_type)
            unique_edge_types = np.unique(edge_type_np)
            hetero[node_type].x = self.x
            for et in unique_edge_types:
                mask = edge_type_np == et
                sub_edge_index = self.edge_index[:, mask]
                hetero[node_type, str(et), node_type].edge_index = sub_edge_index
                if "edge_attr" in self and self.edge_attr is not None:
                    hetero[node_type, str(et), node_type].edge_attr = self.edge_attr[mask]
        else:
            for k in self.node_attrs():
                hetero[node_type][k] = self[k]
            for k in self.edge_attrs():
                hetero[edge_type][k] = self[k]
        return hetero

    def validate(self, raise_on_error: bool = True) -> bool:
        num_nodes = self.num_nodes
        if "edge_index" in self and self.edge_index is not None:
            edge_shape = get_shape(self.edge_index)
            if len(edge_shape) != 2 or edge_shape[0] != 2:
                msg = f"'edge_index' must have shape [2, num_edges], got {edge_shape}"
                if raise_on_error:
                    raise ValueError(msg)
                warnings.warn(msg)
                return False
            if num_nodes is not None and edge_shape[1] > 0:
                edge_max = int(np.max(ops.convert_to_numpy(self.edge_index)))
                if edge_max >= num_nodes:
                    msg = f"'edge_index' references node {edge_max}, but num_nodes is {num_nodes}"
                    if raise_on_error:
                        raise ValueError(msg)
                    warnings.warn(msg)
                    return False
        return True

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        attrs = [size_repr(k, v) for k, v in self._store.items()]
        info = ", ".join(attrs)
        return f"{cls}({info})"
