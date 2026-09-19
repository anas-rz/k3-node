import copy
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from keras import ops

from k3_node.data.data import BaseData, size_repr
from k3_node.data.storage import BaseStorage, GlobalStorage, get_shape, is_tensor_like


class TemporalData(BaseData):
    """A data object composed of a stream of events describing a temporal graph."""

    def __init__(
        self,
        src: Optional[Any] = None,
        dst: Optional[Any] = None,
        t: Optional[Any] = None,
        msg: Optional[Any] = None,
        y: Optional[Any] = None,
        **kwargs,
    ):
        self.__dict__["_store"] = GlobalStorage(_parent=self)
        if src is not None:
            self.src = src
        if dst is not None:
            self.dst = dst
        if t is not None:
            self.t = t
        if msg is not None:
            self.msg = msg
        if y is not None:
            self.y = y
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]) -> "TemporalData":
        return cls(**mapping)

    @property
    def num_events(self) -> int:
        for key in ("src", "dst", "t", "msg"):
            if key in self._store and self._store[key] is not None:
                return get_shape(self._store[key])[0]
        return 0

    @property
    def num_nodes(self) -> int:
        nodes = []
        if "src" in self._store and self._store.src is not None:
            src_np = ops.convert_to_numpy(self._store.src)
            if src_np.size > 0:
                nodes.append(int(np.max(src_np)))
        if "dst" in self._store and self._store.dst is not None:
            dst_np = ops.convert_to_numpy(self._store.dst)
            if dst_np.size > 0:
                nodes.append(int(np.max(dst_np)))
        return max(nodes) + 1 if len(nodes) > 0 else 0

    def __len__(self) -> int:
        return self.num_events

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, str):
            return self._store[idx]
        data = copy.copy(self)
        num_events = self.num_events
        for key, value in data._store.items():
            if is_tensor_like(value) and get_shape(value)[0] == num_events:
                data[key] = value[idx]
        return data

    def __setitem__(self, key: str, value: Any):
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __getattr__(self, key: str) -> Any:
        if "_store" not in self.__dict__:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        try:
            return getattr(self._store, key)
        except AttributeError:
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

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__["_store"] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo=None):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[Any]:
        return [self._store]

    @property
    def edge_stores(self) -> List[Any]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        fields = sorted(list(self.keys()))
        import collections

        TemporalTuple = collections.namedtuple("TemporalTuple", fields)
        return TemporalTuple(**{f: self[f] for f in fields})

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        attrs = [size_repr(k, v) for k, v in self._store.items()]
        return f"{cls}({', '.join(attrs)})"

