import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from k3_node.data import Data, HeteroData
from k3_node.transforms.base_transform import BaseTransform, functional_transform
from k3_node.transforms.utils import as_tensor, is_torch_tensor, match_tensor, to_numpy


@functional_transform("constant")
class Constant(BaseTransform):
    r"""Appends a constant value to each node feature :obj:`x`."""

    def __init__(
        self,
        value: float = 1.0,
        cat: bool = True,
        node_types: Optional[Union[str, List[str]]] = None,
    ):
        if isinstance(node_types, str):
            node_types = [node_types]
        self.value = value
        self.cat = cat
        self.node_types = node_types

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            key = getattr(store, "_key", None)
            if self.node_types is None or key in self.node_types:
                num_nodes = store.num_nodes
                assert num_nodes is not None
                c_np = np.full((num_nodes, 1), self.value, dtype=np.float32)

                if hasattr(store, "x") and store.x is not None and self.cat:
                    x = store.x
                    x_np = to_numpy(x)
                    if x_np.ndim == 1:
                        x_np = x_np.reshape(-1, 1)
                    new_x = np.concatenate([x_np, c_np], axis=-1)
                    store.x = match_tensor(new_x, x)
                else:
                    store.x = match_tensor(c_np, getattr(store, "x", None))

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.value})"


@functional_transform("normalize_features")
class NormalizeFeatures(BaseTransform):
    r"""Row-normalizes node features to sum to 1 (L1-norm)."""

    def __init__(self, attrs: List[str] = ["x"]):
        self.attrs = attrs

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            for key in self.attrs:
                x = store.get(key, None)
                if x is not None:
                    x_np = to_numpy(x).astype(np.float32)
                    if x_np.size > 0:
                        x_np = x_np - np.min(x_np)
                        denom = np.maximum(np.sum(x_np, axis=-1, keepdims=True), 1.0)
                        new_x = x_np / denom
                        store[key] = match_tensor(new_x, x)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attrs={self.attrs})"


@functional_transform("svd_feature_reduction")
class SVDFeatureReduction(BaseTransform):
    r"""Dimensionality reduction of node features via SVD."""

    def __init__(self, out_channels: int):
        self.out_channels = out_channels

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if hasattr(store, "x") and store.x is not None:
                x_np = to_numpy(store.x).astype(np.float32)
                u, s, _ = np.linalg.svd(x_np, full_matrices=False)
                reduced = u[:, : self.out_channels] * s[: self.out_channels]
                store.x = match_tensor(reduced, store.x)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(out_channels={self.out_channels})"


@functional_transform("remove_training_classes")
class RemoveTrainingClasses(BaseTransform):
    r"""Removes training classes from ground-truth labels."""

    def __init__(self, classes: List[int]):
        self.classes = set(classes)

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            if hasattr(store, "y") and store.y is not None:
                y_np = to_numpy(store.y)
                mask = np.isin(y_np, list(self.classes))
                y_np = np.where(mask, -1, y_np)
                store.y = match_tensor(y_np, store.y)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(classes={sorted(list(self.classes))})"


@functional_transform("random_node_split")
class RandomNodeSplit(BaseTransform):
    r"""Performs a random node-level train/val/test split."""

    def __init__(
        self,
        split: str = "train_rest",
        num_splits: int = 1,
        num_train_per_class: int = 20,
        num_val: Union[int, float] = 500,
        num_test: Union[int, float] = 1000,
        key: Optional[str] = "y",
    ):
        self.split = split
        self.num_splits = num_splits
        self.num_train_per_class = num_train_per_class
        self.num_val = num_val
        self.num_test = num_test
        self.key = key

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            num_nodes = store.num_nodes
            assert num_nodes is not None

            train_masks, val_masks, test_masks = [], [], []
            for _ in range(self.num_splits):
                train_mask = np.zeros(num_nodes, dtype=bool)
                val_mask = np.zeros(num_nodes, dtype=bool)
                test_mask = np.zeros(num_nodes, dtype=bool)

                if self.split == "random":
                    perm = np.random.permutation(num_nodes)
                    n_val = int(self.num_val * num_nodes) if isinstance(self.num_val, float) else self.num_val
                    n_test = int(self.num_test * num_nodes) if isinstance(self.num_test, float) else self.num_test
                    n_train = num_nodes - n_val - n_test

                    train_mask[perm[:n_train]] = True
                    val_mask[perm[n_train : n_train + n_val]] = True
                    test_mask[perm[n_train + n_val :]] = True
                elif self.split == "test_rest":
                    perm = np.random.permutation(num_nodes)
                    n_val = int(self.num_val * num_nodes) if isinstance(self.num_val, float) else self.num_val
                    n_train = self.num_train_per_class
                    train_mask[perm[:n_train]] = True
                    val_mask[perm[n_train : n_train + n_val]] = True
                    test_mask[perm[n_train + n_val :]] = True
                else:  # train_rest
                    perm = np.random.permutation(num_nodes)
                    n_val = int(self.num_val * num_nodes) if isinstance(self.num_val, float) else self.num_val
                    n_test = int(self.num_test * num_nodes) if isinstance(self.num_test, float) else self.num_test
                    val_mask[perm[:n_val]] = True
                    test_mask[perm[n_val : n_val + n_test]] = True
                    train_mask[perm[n_val + n_test :]] = True

                train_masks.append(train_mask)
                val_masks.append(val_mask)
                test_masks.append(test_mask)

            ref = getattr(store, "x", getattr(store, "y", getattr(store, "edge_index", getattr(store, "pos", None))))
            if self.num_splits == 1:
                store.train_mask = match_tensor(train_masks[0], ref, dtype="bool")
                store.val_mask = match_tensor(val_masks[0], ref, dtype="bool")
                store.test_mask = match_tensor(test_masks[0], ref, dtype="bool")
            else:
                store.train_mask = match_tensor(np.stack(train_masks, axis=-1), ref, dtype="bool")
                store.val_mask = match_tensor(np.stack(val_masks, axis=-1), ref, dtype="bool")
                store.test_mask = match_tensor(np.stack(test_masks, axis=-1), ref, dtype="bool")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(split={self.split}, num_splits={self.num_splits})"


@functional_transform("random_link_split")
class RandomLinkSplit(BaseTransform):
    r"""Performs an edge-level random split into train, val, and test edges."""

    def __init__(
        self,
        num_val: float = 0.1,
        num_test: float = 0.2,
        is_undirected: bool = False,
        key_negative_edges: Optional[str] = None,
        split_labels: bool = False,
        add_negative_train_samples: bool = False,
        neg_sampling_ratio: float = 1.0,
        disjoint_train_ratio: float = 0.0,
        edge_types: Optional[List[Any]] = None,
        rev_edge_types: Optional[List[Any]] = None,
    ):
        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = is_undirected
        self.key_negative_edges = key_negative_edges
        self.split_labels = split_labels
        self.add_negative_train_samples = add_negative_train_samples
        self.neg_sampling_ratio = neg_sampling_ratio
        self.disjoint_train_ratio = disjoint_train_ratio
        self.edge_types = edge_types
        self.rev_edge_types = rev_edge_types

    def forward(self, data: Union[Data, HeteroData]) -> Tuple[Any, Any, Any]:
        train_data = copy.copy(data)
        val_data = copy.copy(data)
        test_data = copy.copy(data)

        if isinstance(data, Data):
            edge_index = to_numpy(data.edge_index)
            num_edges = edge_index.shape[1]
            if self.is_undirected:
                mask = edge_index[0] <= edge_index[1]
                perm = np.where(mask)[0]
                perm = perm[np.random.permutation(len(perm))]
            else:
                perm = np.random.permutation(num_edges)

            num_total = len(perm)
            n_val = int(self.num_val * num_total)
            n_test = int(self.num_test * num_total)
            n_train = num_total - n_val - n_test

            train_idx = perm[:n_train]
            val_idx = perm[n_train : n_train + n_val]
            test_idx = perm[n_train + n_val :]
            train_val_idx = perm[: n_train + n_val]

            def to_edges(idx, undirected=False):
                edges = edge_index[:, idx]
                if undirected:
                    edges = np.concatenate([edges, edges[::-1]], axis=1)
                return edges

            train_data.edge_index = match_tensor(to_edges(train_idx, self.is_undirected), data.edge_index)
            val_data.edge_index = train_data.edge_index
            test_data.edge_index = match_tensor(to_edges(train_val_idx, self.is_undirected), data.edge_index)

            from k3_node.models.utils import negative_sampling
            num_nodes = data.num_nodes or (int(np.max(edge_index)) + 1 if edge_index.size > 0 else 0)
            num_neg_val = n_val
            num_neg_test = n_test
            num_neg_train = n_train if self.add_negative_train_samples else 0
            total_neg = max(num_neg_val + num_neg_test + num_neg_train, 1)
            neg_all = negative_sampling(data.edge_index, num_nodes=num_nodes, num_neg_samples=total_neg)
            neg_all_np = to_numpy(neg_all)

            neg_val = neg_all_np[:, :num_neg_val]
            neg_test = neg_all_np[:, num_neg_val : num_neg_val + num_neg_test]
            neg_train = neg_all_np[:, num_neg_val + num_neg_test :] if num_neg_train > 0 else None

            if self.split_labels:
                train_data.pos_edge_label_index = match_tensor(edge_index[:, train_idx], data.edge_index)
                train_data.pos_edge_label = match_tensor(np.ones(train_idx.shape[0], dtype=np.float32), data.edge_index)

                val_data.pos_edge_label_index = match_tensor(edge_index[:, val_idx], data.edge_index)
                val_data.pos_edge_label = match_tensor(np.ones(val_idx.shape[0], dtype=np.float32), data.edge_index)
                val_data.neg_edge_label_index = match_tensor(neg_val, data.edge_index)
                val_data.neg_edge_label = match_tensor(np.zeros(neg_val.shape[1], dtype=np.float32), data.edge_index)

                test_data.pos_edge_label_index = match_tensor(edge_index[:, test_idx], data.edge_index)
                test_data.pos_edge_label = match_tensor(np.ones(test_idx.shape[0], dtype=np.float32), data.edge_index)
                test_data.neg_edge_label_index = match_tensor(neg_test, data.edge_index)
                test_data.neg_edge_label = match_tensor(np.zeros(neg_test.shape[1], dtype=np.float32), data.edge_index)

                if self.add_negative_train_samples and neg_train is not None:
                    train_data.neg_edge_label_index = match_tensor(neg_train, data.edge_index)
                    train_data.neg_edge_label = match_tensor(np.zeros(neg_train.shape[1], dtype=np.float32), data.edge_index)
            else:
                train_data.edge_label_index = match_tensor(edge_index[:, train_idx], data.edge_index)
                train_data.edge_label = match_tensor(np.ones(train_idx.shape[0], dtype=np.float32), data.edge_index)

                val_data.edge_label_index = match_tensor(edge_index[:, val_idx], data.edge_index)
                val_data.edge_label = match_tensor(np.ones(val_idx.shape[0], dtype=np.float32), data.edge_index)

                test_data.edge_label_index = match_tensor(edge_index[:, test_idx], data.edge_index)
                test_data.edge_label = match_tensor(np.ones(test_idx.shape[0], dtype=np.float32), data.edge_index)

        return train_data, val_data, test_data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_val={self.num_val}, num_test={self.num_test})"


@functional_transform("node_property_split")
class NodePropertySplit(BaseTransform):
    r"""Splits nodes based on an ordered node property."""

    def __init__(
        self,
        node_property: Union[str, Any],
        num_splits: int = 1,
        num_val: Union[int, float] = 0.1,
        num_test: Union[int, float] = 0.2,
        ascending: bool = True,
    ):
        self.node_property = node_property
        self.num_splits = num_splits
        self.num_val = num_val
        self.num_test = num_test
        self.ascending = ascending

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.node_stores:
            num_nodes = store.num_nodes
            prop = store[self.node_property] if isinstance(self.node_property, str) else self.node_property
            prop_np = to_numpy(prop).reshape(-1)

            order = np.argsort(prop_np)
            if not self.ascending:
                order = order[::-1]

            n_val = int(self.num_val * num_nodes) if isinstance(self.num_val, float) else self.num_val
            n_test = int(self.num_test * num_nodes) if isinstance(self.num_test, float) else self.num_test
            n_train = num_nodes - n_val - n_test

            train_mask = np.zeros(num_nodes, dtype=bool)
            val_mask = np.zeros(num_nodes, dtype=bool)
            test_mask = np.zeros(num_nodes, dtype=bool)

            train_mask[order[:n_train]] = True
            val_mask[order[n_train : n_train + n_val]] = True
            test_mask[order[n_train + n_val :]] = True

            store.train_mask = match_tensor(train_mask, getattr(store, "x", None), dtype="bool")
            store.val_mask = match_tensor(val_mask, getattr(store, "x", None), dtype="bool")
            store.test_mask = match_tensor(test_mask, getattr(store, "x", None), dtype="bool")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_val={self.num_val}, num_test={self.num_test})"


@functional_transform("index_to_mask")
class IndexToMask(BaseTransform):
    r"""Converts node or edge indices to a boolean mask representation."""

    def __init__(
        self,
        attrs: Optional[Union[str, List[str]]] = None,
        sizes: Optional[Union[int, List[int]]] = None,
        replace: bool = False,
    ):
        self.attrs = [attrs] if isinstance(attrs, str) else attrs
        self.sizes = sizes
        self.replace = replace

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.stores:
            attrs = self.attrs or [k for k in store.keys() if k.endswith("_index") and k != "edge_index"]
            for attr in attrs:
                if attr not in store or attr == "edge_index":
                    continue
                idx_np = to_numpy(store[attr]).astype(np.int64)
                size = self.sizes if isinstance(self.sizes, int) else None
                if size is None:
                    size = int(np.max(idx_np)) + 1 if idx_np.size > 0 else 0
                    if store.is_edge_attr(attr) and store.num_edges is not None:
                        size = max(size, store.num_edges)
                    elif store.num_nodes is not None:
                        size = max(size, store.num_nodes)

                mask = np.zeros(size, dtype=bool)
                mask[idx_np] = True
                mask_key = f"{attr[:-6]}_mask" if attr.endswith("_index") else f"{attr}_mask"
                store[mask_key] = match_tensor(mask, store[attr], dtype="bool")
                if self.replace:
                    del store[attr]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attrs={self.attrs}, replace={self.replace})"


@functional_transform("mask_to_index")
class MaskToIndex(BaseTransform):
    r"""Converts boolean masks to indices."""

    def __init__(
        self,
        attrs: Optional[Union[str, List[str]]] = None,
        replace: bool = False,
    ):
        self.attrs = [attrs] if isinstance(attrs, str) else attrs
        self.replace = replace

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.stores:
            attrs = self.attrs or [k for k in store.keys() if k.endswith("_mask")]
            for attr in attrs:
                if attr not in store:
                    continue
                mask_np = to_numpy(store[attr]).astype(bool)
                idx_np = np.nonzero(mask_np)[0].astype(np.int64)
                idx_key = f"{attr[:-5]}_index" if attr.endswith("_mask") else f"{attr}_index"
                store[idx_key] = match_tensor(idx_np, store[attr], dtype="int64")
                if self.replace:
                    del store[attr]

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(attrs={self.attrs}, replace={self.replace})"


class Padding(ABC):
    r"""Abstract class for specifying padding values."""

    @abstractmethod
    def get_value(self, store_type: Optional[Any] = None, attr_name: Optional[str] = None) -> Union[int, float]:
        pass


@dataclass(init=False)
class UniformPadding(Padding):
    r"""Uniform padding with a constant value."""

    value: Union[int, float] = 0.0

    def __init__(self, value: Union[int, float] = 0.0):
        self.value = value

    def get_value(self, store_type: Optional[Any] = None, attr_name: Optional[str] = None) -> Union[int, float]:
        return self.value


@dataclass(init=False)
class MappingPadding(Padding):
    r"""Mapping padding with attribute-specific padding values."""

    values: Dict[Any, Any]
    default: UniformPadding

    def __init__(self, values: Dict[Any, Union[int, float, Padding]], default: Union[int, float] = 0.0):
        self.values = values
        self.default = UniformPadding(default)

    def get_value(self, store_type: Optional[Any] = None, attr_name: Optional[str] = None) -> Union[int, float]:
        val = self.values.get(attr_name, self.values.get(store_type, self.default))
        if isinstance(val, Padding):
            return val.get_value(store_type, attr_name)
        return val


@functional_transform("pad")
class Pad(BaseTransform):
    r"""Pads node and edge features to a maximum number of nodes and edges."""

    def __init__(
        self,
        max_num_nodes: Optional[int] = None,
        max_num_edges: Optional[int] = None,
        node_padding: Union[int, float, Padding] = 0.0,
        edge_padding: Union[int, float, Padding] = 0.0,
    ):
        self.max_num_nodes = max_num_nodes
        self.max_num_edges = max_num_edges
        self.node_padding = node_padding if isinstance(node_padding, Padding) else UniformPadding(node_padding)
        self.edge_padding = edge_padding if isinstance(edge_padding, Padding) else UniformPadding(edge_padding)

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        orig_num_nodes = data.num_nodes
        for store in data.node_stores:
            if self.max_num_nodes is not None and store.num_nodes is not None:
                pad_nodes = self.max_num_nodes - store.num_nodes
                if pad_nodes > 0:
                    for key, val in list(store.items()):
                        if store.is_node_attr(key):
                            val_np = to_numpy(val)
                            pad_shape = (pad_nodes,) + val_np.shape[1:]
                            pad_val = self.node_padding.get_value(getattr(store, "_key", None), key)
                            padding = np.full(pad_shape, pad_val, dtype=val_np.dtype)
                            store[key] = match_tensor(np.concatenate([val_np, padding], axis=0), val)
                    store.num_nodes = self.max_num_nodes

        max_num_edges = self.max_num_edges
        if max_num_edges is None and self.max_num_nodes is not None:
            max_num_edges = self.max_num_nodes * self.max_num_nodes

        for store in data.edge_stores:
            if max_num_edges is not None and store.num_edges is not None:
                pad_edges = max_num_edges - store.num_edges
                if pad_edges > 0:
                    if "edge_index" in store and store.edge_index is not None:
                        ei_np = to_numpy(store.edge_index)
                        pad_val = orig_num_nodes if orig_num_nodes is not None else 0
                        padding_ei = np.full((2, pad_edges), pad_val, dtype=ei_np.dtype)
                        store.edge_index = match_tensor(np.concatenate([ei_np, padding_ei], axis=1), store.edge_index)
                    for key, val in list(store.items()):
                        if store.is_edge_attr(key) and key != "edge_index":
                            val_np = to_numpy(val)
                            pad_shape = (pad_edges,) + val_np.shape[1:]
                            pad_val = self.edge_padding.get_value(getattr(store, "_key", None), key)
                            padding = np.full(pad_shape, pad_val, dtype=val_np.dtype)
                            store[key] = match_tensor(np.concatenate([val_np, padding], axis=0), val)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_num_nodes={self.max_num_nodes}, max_num_edges={self.max_num_edges})"


@functional_transform("to_device")
class ToDevice(BaseTransform):
    r"""Performs tensor device conversion."""

    def __init__(self, device: Union[int, str], attrs: Optional[List[str]] = None, non_blocking: bool = False):
        self.device = device
        self.attrs = attrs or []
        self.non_blocking = non_blocking

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        if hasattr(data, "to"):
            return data.to(self.device, *self.attrs, non_blocking=self.non_blocking)
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.device})"


@functional_transform("to_sparse_tensor")
class ToSparseTensor(BaseTransform):
    r"""Converts edge_index into a sparse adjacency representation."""

    def __init__(
        self,
        attr: Optional[str] = "edge_weight",
        remove_edge_index: bool = True,
        fill_cache: bool = True,
        layout: Optional[int] = None,
    ):
        self.attr = attr
        self.remove_edge_index = remove_edge_index
        self.fill_cache = fill_cache
        self.layout = layout

    def forward(self, data: Union[Data, HeteroData]) -> Union[Data, HeteroData]:
        for store in data.edge_stores:
            if "edge_index" not in store:
                continue
            ei = store.edge_index
            val = store.get(self.attr, None)
            ei_np = to_numpy(ei)
            num_nodes = store.size(0) if hasattr(store, "size") else None
            if num_nodes is None:
                num_nodes = int(np.max(ei_np)) + 1 if ei_np.size > 0 else 0

            # Store adjacency
            store.adj_t = ei
            if self.remove_edge_index:
                del store["edge_index"]
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
