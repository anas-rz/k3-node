import copy
from collections import defaultdict
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from keras import ops

from k3_node.data.data import BaseData, Data, size_repr
from k3_node.data.storage import (
    BaseStorage,
    EdgeStorage,
    NodeStorage,
    get_shape,
    is_tensor_like,
)

NodeType = str
EdgeType = Tuple[str, str, str]


class HeteroData(BaseData):
    """A data object describing a heterogeneous graph."""

    def __init__(self, _mapping: Optional[Dict[str, Any]] = None, **kwargs):
        self.__dict__["_node_store_dict"] = {}
        self.__dict__["_edge_store_dict"] = {}

        for key, value in (_mapping or {}).items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _to_edge_type(self, key: Any) -> Optional[EdgeType]:
        if isinstance(key, tuple):
            if len(key) == 3:
                return (str(key[0]), str(key[1]), str(key[2]))
            if len(key) == 2:
                # Find matching edge type
                matches = [k for k in self.edge_types if k[0] == key[0] and k[-1] == key[1]]
                if len(matches) == 1:
                    return matches[0]
                elif len(matches) > 1:
                    raise KeyError(f"Ambiguous edge type for '{key}': {matches}")
                return (str(key[0]), "to", str(key[1]))
        elif isinstance(key, str):
            matches = [k for k in self.edge_types if k[1] == key]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                raise KeyError(f"Ambiguous edge type for rel '{key}': {matches}")
        return None

    def __getitem__(self, key: Any) -> Any:
        edge_key = self._to_edge_type(key)
        if edge_key is not None and (edge_key in self._edge_store_dict or isinstance(key, tuple)):
            if edge_key not in self._edge_store_dict:
                store = EdgeStorage(_parent=self)
                store.__dict__["_key"] = edge_key
                self._edge_store_dict[edge_key] = store
            return self._edge_store_dict[edge_key]

        if isinstance(key, str):
            if key not in self._node_store_dict:
                store = NodeStorage(_parent=self)
                store.__dict__["_key"] = key
                self._node_store_dict[key] = store
            return self._node_store_dict[key]

        raise KeyError(f"Invalid key '{key}' for {self.__class__.__name__}")

    def __setitem__(self, key: Any, value: Any):
        store = self[key]
        if isinstance(value, BaseStorage):
            for k, v in value.items():
                store[k] = v
        elif isinstance(value, dict):
            for k, v in value.items():
                store[k] = v
        else:
            raise ValueError(f"Value for key '{key}' must be a Storage or dict, got {type(value)}")

    def __delitem__(self, key: Any):
        edge_key = self._to_edge_type(key)
        if edge_key is not None and edge_key in self._edge_store_dict:
            del self._edge_store_dict[edge_key]
        elif isinstance(key, str) and key in self._node_store_dict:
            del self._node_store_dict[key]
        else:
            raise KeyError(key)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__:
            return self.__dict__[key]
        if "_node_store_dict" in self.__dict__ and key in self._node_store_dict:
            return self._node_store_dict[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any):
        if key in ("_node_store_dict", "_edge_store_dict"):
            self.__dict__[key] = value
        elif isinstance(value, BaseStorage):
            if isinstance(value, NodeStorage):
                self._node_store_dict[key] = value
            elif isinstance(value, EdgeStorage):
                edge_key = self._to_edge_type(key)
                self._edge_store_dict[edge_key or key] = value
        else:
            self.__dict__[key] = value

    @property
    def node_types(self) -> List[NodeType]:
        return list(self._node_store_dict.keys())

    @property
    def edge_types(self) -> List[EdgeType]:
        return list(self._edge_store_dict.keys())

    def metadata(self) -> Tuple[List[NodeType], List[EdgeType]]:
        return self.node_types, self.edge_types

    @property
    def stores(self) -> List[BaseStorage]:
        return list(self._node_store_dict.values()) + list(self._edge_store_dict.values())

    @property
    def node_stores(self) -> List[NodeStorage]:
        return list(self._node_store_dict.values())

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return list(self._edge_store_dict.values())

    @property
    def num_nodes_dict(self) -> Dict[NodeType, int]:
        return {k: v.num_nodes for k, v in self._node_store_dict.items()}

    @property
    def num_edges_dict(self) -> Dict[EdgeType, int]:
        return {k: v.num_edges for k, v in self._edge_store_dict.items()}

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        out.__dict__["_node_store_dict"] = {}
        out.__dict__["_edge_store_dict"] = {}
        for k, v in self._node_store_dict.items():
            store = copy.copy(v)
            setattr(store, "_parent", out)
            out._node_store_dict[k] = store
        for k, v in self._edge_store_dict.items():
            store = copy.copy(v)
            setattr(store, "_parent", out)
            out._edge_store_dict[k] = store
        return out

    def __deepcopy__(self, memo=None):
        out = self.__class__.__new__(self.__class__)
        out.__dict__["_node_store_dict"] = {}
        out.__dict__["_edge_store_dict"] = {}
        for k, v in self._node_store_dict.items():
            store = copy.deepcopy(v, memo)
            setattr(store, "_parent", out)
            out._node_store_dict[k] = store
        for k, v in self._edge_store_dict.items():
            store = copy.deepcopy(v, memo)
            setattr(store, "_parent", out)
            out._edge_store_dict[k] = store
        return out

    def __getstate__(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, mapping: Dict[str, Any]):
        import weakref

        for key, value in mapping.items():
            self.__dict__[key] = value
        for store in self.stores:
            store.__dict__["_parent"] = weakref.ref(self)

    def clone(self) -> "HeteroData":
        return copy.deepcopy(self)

    def collect(self, key: str, allow_missing: bool = True) -> Dict[Any, Any]:
        mapping = {}
        for k, store in list(self._node_store_dict.items()) + list(self._edge_store_dict.items()):
            if key in store:
                mapping[k] = store[key]
            elif not allow_missing:
                raise KeyError(f"Key '{key}' not found in store '{k}'")
        return mapping

    def __inc__(self, key: str, value: Any, store: Optional[BaseStorage] = None, *args, **kwargs) -> Any:
        if "batch" in key:
            return int(value.max()) + 1 if is_tensor_like(value) and value.ndim > 0 and value.shape[0] > 0 else 0
        if "index" in key and store is not None and isinstance(store, EdgeStorage):
            edge_type = store._key
            src, _, dst = edge_type
            src_num = self[src].num_nodes
            dst_num = self[dst].num_nodes
            return np.array([[src_num], [dst_num]])
        return 0

    def __cat_dim__(self, key: str, value: Any, store: Optional[BaseStorage] = None, *args, **kwargs) -> int:
        if key in ("edge_index", "adj_t"):
            return -1
        if is_tensor_like(value) and len(get_shape(value)) == 2 and get_shape(value)[0] == 2 and "index" in key:
            return -1
        return 0

    def to_dict(self) -> Dict[str, Any]:
        out = {}
        for k, store in self._node_store_dict.items():
            out[k] = store.to_dict()
        for k, store in self._edge_store_dict.items():
            out[k] = store.to_dict()
        return out

    def to_namedtuple(self) -> NamedTuple:
        # Build nested namedtuple
        node_fields = sorted(list(self._node_store_dict.keys()))
        node_dict = {k: self._node_store_dict[k].to_dict() for k in node_fields}
        edge_fields = [f"{k[0]}__{k[1]}__{k[2]}" for k in sorted(list(self._edge_store_dict.keys()))]
        edge_dict = {f"{k[0]}__{k[1]}__{k[2]}": self._edge_store_dict[k].to_dict() for k in sorted(list(self._edge_store_dict.keys()))}
        all_fields = node_fields + edge_fields
        HeteroTuple = collections.namedtuple("HeteroTuple", all_fields)
        return HeteroTuple(**node_dict, **edge_dict)

    def edge_type_subgraph(self, edge_types: List[EdgeType]) -> "HeteroData":
        out = copy.deepcopy(self)
        for et in list(out._edge_store_dict.keys()):
            if et not in edge_types:
                del out._edge_store_dict[et]
        return out

    def subgraph(self, subset_dict: Dict[NodeType, Any]) -> "HeteroData":
        out = copy.deepcopy(self)
        for node_type, subset in subset_dict.items():
            store = out[node_type]
            subset_np = ops.convert_to_numpy(subset)
            indices = np.where(subset_np)[0] if subset_np.dtype == bool else subset_np
            for key in store.node_attrs():
                val = store[key]
                if is_tensor_like(val):
                    store[key] = ops.take(val, indices, axis=self.__cat_dim__(key, val, store))
        return out

    def to_homogeneous(
        self,
        node_attrs: Optional[List[str]] = None,
        edge_attrs: Optional[List[str]] = None,
        add_node_type: bool = True,
        add_edge_type: bool = True,
        dummy_values: bool = True,
    ) -> Data:
        data = Data()

        # Compute node offsets and slices
        node_slices = {}
        curr_offset = 0
        node_type_list = []
        for i, node_type in enumerate(self.node_types):
            num_nodes = self[node_type].num_nodes
            node_slices[node_type] = curr_offset
            if add_node_type:
                node_type_list.append(np.full((num_nodes,), i, dtype=np.int64))
            curr_offset += num_nodes

        data.num_nodes = curr_offset
        if add_node_type and len(node_type_list) > 0:
            node_type_arr = np.concatenate(node_type_list, axis=0)
            data.node_type = ops.convert_to_tensor(node_type_arr, dtype="int64")

        # Concat node features
        if node_attrs is None:
            # find common node attrs across node types
            all_node_attrs = set()
            for store in self.node_stores:
                all_node_attrs.update(store.node_attrs())
            node_attrs = list(all_node_attrs)

        for attr in node_attrs:
            attr_vals = []
            for node_type in self.node_types:
                val = self[node_type].get(attr)
                if val is not None:
                    attr_vals.append(ops.convert_to_numpy(val))
                elif dummy_values:
                    num_nodes = self[node_type].num_nodes
                    attr_vals.append(np.zeros((num_nodes, 0), dtype=np.float32))
            if len(attr_vals) > 0:
                concat_val = np.concatenate(attr_vals, axis=0)
                data[attr] = ops.convert_to_tensor(concat_val)

        # Offsetting edge indices
        edge_indices = []
        edge_type_list = []
        for i, edge_type in enumerate(self.edge_types):
            store = self[edge_type]
            edge_index = store.get("edge_index")
            if edge_index is not None:
                ei_np = ops.convert_to_numpy(edge_index).copy()
                src_offset = node_slices[edge_type[0]]
                dst_offset = node_slices[edge_type[-1]]
                ei_np[0] += src_offset
                ei_np[1] += dst_offset
                edge_indices.append(ei_np)
                if add_edge_type:
                    edge_type_list.append(np.full((ei_np.shape[1],), i, dtype=np.int64))

        if len(edge_indices) > 0:
            concat_ei = np.concatenate(edge_indices, axis=1)
            data.edge_index = ops.convert_to_tensor(concat_ei, dtype="int64")
        if add_edge_type and len(edge_type_list) > 0:
            concat_et = np.concatenate(edge_type_list, axis=0)
            data.edge_type = ops.convert_to_tensor(concat_et, dtype="int64")

        return data

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info_lines = []
        for k, store in self._node_store_dict.items():
            attrs = [size_repr(attr, store[attr]) for attr in store.keys()]
            info_lines.append(f"  {k}={{{', '.join(attrs)}}}")
        for k, store in self._edge_store_dict.items():
            attrs = [size_repr(attr, store[attr]) for attr in store.keys()]
            edge_name = f"('{k[0]}', '{k[1]}', '{k[2]}')"
            info_lines.append(f"  {edge_name}={{{', '.join(attrs)}}}")
        info = ",\n".join(info_lines)
        return f"{cls}(\n{info}\n)" if info else f"{cls}()"
