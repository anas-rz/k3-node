import random
from collections import defaultdict
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from keras import ops

from k3_node.data import Data, HeteroData, InMemoryDataset
from k3_node.layers.conv.utils import remove_self_loops
from k3_node.transforms.utils import to_undirected
from k3_node.utils.graph import coalesce


def get_num_nodes(avg_num_nodes: int, avg_degree: float) -> int:
    min_num_nodes = max(3 * avg_num_nodes // 4, int(avg_degree))
    max_num_nodes = 5 * avg_num_nodes // 4
    return random.randint(min_num_nodes, max_num_nodes)


def get_num_channels(num_channels: int) -> int:
    min_num_channels = 3 * num_channels // 4
    max_num_channels = 5 * num_channels // 4
    return random.randint(min_num_channels, max_num_channels)


def get_edge_index(
    num_src_nodes: int,
    num_dst_nodes: int,
    avg_degree: float,
    is_undirected: bool = False,
    remove_loops: bool = False,
):
    num_edges = int(num_src_nodes * avg_degree)
    row = np.random.randint(0, num_src_nodes, size=(num_edges,), dtype=np.int64)
    col = np.random.randint(0, num_dst_nodes, size=(num_edges,), dtype=np.int64)
    edge_index = np.stack([row, col], axis=0)

    if remove_loops:
        edge_index, _ = remove_self_loops(edge_index)

    num_nodes = max(num_src_nodes, num_dst_nodes)
    if is_undirected:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    else:
        edge_index, _ = coalesce(edge_index, num_nodes=num_nodes)

    return ops.convert_to_tensor(edge_index, dtype="int64")


class FakeDataset(InMemoryDataset):
    r"""A fake dataset that returns randomly generated `k3_node.data.Data` objects."""

    def __init__(
        self,
        num_graphs: int = 1,
        avg_num_nodes: int = 1000,
        avg_degree: float = 10.0,
        num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: str = "auto",
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs: Union[int, Tuple[int, ...]],
    ):
        super().__init__(None, transform)

        if task == "auto":
            task = "graph" if num_graphs > 1 else "node"
        assert task in ["node", "graph"]

        self.num_graphs_val = num_graphs
        self.avg_num_nodes = max(avg_num_nodes, int(avg_degree))
        self.avg_degree = max(avg_degree, 1)
        self.num_channels = num_channels
        self.edge_dim = edge_dim
        self._num_classes = num_classes
        self.task = task
        self.is_undirected = is_undirected
        self.kwargs = kwargs

        data_list = [self.generate_data() for _ in range(max(num_graphs, 1))]
        self.data, self.slices = self.collate(data_list)

    def __repr__(self) -> str:
        return f"FakeDataset({self.num_graphs_val})" if self.num_graphs_val > 1 else "FakeDataset()"

    def generate_data(self) -> Data:
        num_nodes = get_num_nodes(self.avg_num_nodes, self.avg_degree)
        data = Data()

        if self._num_classes > 0 and self.task == "node":
            data.y = ops.convert_to_tensor(
                np.random.randint(0, self._num_classes, size=(num_nodes,), dtype=np.int64),
                dtype="int64",
            )
        elif self._num_classes > 0 and self.task == "graph":
            data.y = ops.convert_to_tensor(
                np.array([random.randint(0, self._num_classes - 1)], dtype=np.int64),
                dtype="int64",
            )

        data.edge_index = get_edge_index(
            num_nodes, num_nodes, self.avg_degree, self.is_undirected, remove_loops=True
        )

        if self.num_channels > 0:
            x = np.random.randn(num_nodes, self.num_channels).astype(np.float32)
            if self._num_classes > 0 and self.task == "node":
                y_np = ops.convert_to_numpy(data.y)
                x = x + y_np[:, None]
            elif self._num_classes > 0 and self.task == "graph":
                y_np = ops.convert_to_numpy(data.y)
                x = x + y_np
            data.x = ops.convert_to_tensor(x, dtype="float32")
        else:
            data.num_nodes = num_nodes

        num_edges = int(ops.shape(data.edge_index)[1])
        if self.edge_dim > 1:
            data.edge_attr = ops.convert_to_tensor(
                np.random.rand(num_edges, self.edge_dim).astype(np.float32),
                dtype="float32",
            )
        elif self.edge_dim == 1:
            data.edge_weight = ops.convert_to_tensor(
                np.random.rand(num_edges).astype(np.float32),
                dtype="float32",
            )

        for feature_name, feature_shape in self.kwargs.items():
            shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
            setattr(
                data,
                feature_name,
                ops.convert_to_tensor(np.random.randn(*shape).astype(np.float32), dtype="float32"),
            )

        return data


class FakeHeteroDataset(InMemoryDataset):
    r"""A fake dataset that returns randomly generated `k3_node.data.HeteroData` objects."""

    def __init__(
        self,
        num_graphs: int = 1,
        num_node_types: int = 3,
        num_edge_types: int = 6,
        avg_num_nodes: int = 1000,
        avg_degree: float = 10.0,
        avg_num_channels: int = 64,
        edge_dim: int = 0,
        num_classes: int = 10,
        task: str = "auto",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs: Union[int, Tuple[int, ...]],
    ):
        super().__init__(None, transform)

        if task == "auto":
            task = "graph" if num_graphs > 1 else "node"
        assert task in ["node", "graph"]

        self.num_graphs_val = num_graphs
        self.node_types = [f"v{i}" for i in range(max(num_node_types, 1))]

        edge_types: List[Tuple[str, str]] = []
        edge_type_product = list(product(self.node_types, self.node_types))
        while len(edge_types) < max(num_edge_types, 1):
            edge_types.extend(edge_type_product)
        random.shuffle(edge_types)

        self.edge_types: List[Tuple[str, str, str]] = []
        count: Dict[Tuple[str, str], int] = defaultdict(int)
        for edge_type in edge_types[: max(num_edge_types, 1)]:
            rel = f"e{count[edge_type]}"
            count[edge_type] += 1
            self.edge_types.append((edge_type[0], rel, edge_type[1]))

        self.avg_num_nodes = max(avg_num_nodes, int(avg_degree))
        self.avg_degree = max(avg_degree, 1)
        self.avg_num_channels = avg_num_channels
        self.edge_dim = edge_dim
        self._num_classes = num_classes
        self.task = task
        self.kwargs = kwargs

        data_list = [self.generate_data() for _ in range(max(num_graphs, 1))]
        self.data, self.slices = self.collate(data_list)

    def __repr__(self) -> str:
        return f"FakeHeteroDataset({self.num_graphs_val})" if self.num_graphs_val > 1 else "FakeHeteroDataset()"

    def generate_data(self) -> HeteroData:
        data = HeteroData()

        for node_type in self.node_types:
            num_nodes = get_num_nodes(self.avg_num_nodes, self.avg_degree)
            num_channels = get_num_channels(self.avg_num_channels)
            store = data[node_type]

            if self.avg_num_channels > 0:
                store.x = ops.convert_to_tensor(
                    np.random.randn(num_nodes, num_channels).astype(np.float32),
                    dtype="float32",
                )
            else:
                store.num_nodes = num_nodes

            if self._num_classes > 0 and self.task == "node":
                store.y = ops.convert_to_tensor(
                    np.random.randint(0, self._num_classes, size=(num_nodes,), dtype=np.int64),
                    dtype="int64",
                )

        for edge_type in self.edge_types:
            src, rel, dst = edge_type
            store = data[edge_type]
            store.edge_index = get_edge_index(
                data[src].num_nodes,
                data[dst].num_nodes,
                self.avg_degree,
                is_undirected=False,
                remove_loops=False,
            )

            num_edges = int(ops.shape(store.edge_index)[1])
            if self.edge_dim > 1:
                store.edge_attr = ops.convert_to_tensor(
                    np.random.rand(num_edges, self.edge_dim).astype(np.float32),
                    dtype="float32",
                )
            elif self.edge_dim == 1:
                store.edge_weight = ops.convert_to_tensor(
                    np.random.rand(num_edges).astype(np.float32),
                    dtype="float32",
                )

        if self._num_classes > 0 and self.task == "graph":
            data.y = ops.convert_to_tensor(
                np.array([random.randint(0, self._num_classes - 1)], dtype=np.int64),
                dtype="int64",
            )

        for feature_name, feature_shape in self.kwargs.items():
            shape = (feature_shape,) if isinstance(feature_shape, int) else feature_shape
            setattr(
                data,
                feature_name,
                ops.convert_to_tensor(np.random.randn(*shape).astype(np.float32), dtype="float32"),
            )

        return data
