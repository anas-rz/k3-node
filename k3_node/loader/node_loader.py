from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

try:
    import torch
    from torch import Tensor
except ImportError:
    torch = None
    Tensor = type(None)

from k3_node.data import Data, HeteroData
from k3_node.loader.base import BaseDataLoader, DataLoaderIterator
from k3_node.loader.mixin import AffinityMixin, LogMemoryMixin, MultithreadingMixin
from k3_node.loader.utils import (
    filter_data,
    filter_hetero_data,
    get_input_nodes,
    infer_filter_per_worker,
)


@dataclass
class NodeSamplerInput:
    input_id: Optional[Any]
    node: Any
    time: Optional[Any] = None
    input_type: Optional[str] = None

    def __getitem__(self, index: Any) -> 'NodeSamplerInput':
        if torch is not None and not isinstance(index, Tensor):
            index = torch.as_tensor(index, dtype=torch.long)
        return NodeSamplerInput(
            input_id=self.input_id[index] if self.input_id is not None else index,
            node=self.node[index],
            time=self.time[index] if self.time is not None else None,
            input_type=self.input_type,
        )


@dataclass
class SamplerOutput:
    node: Any
    row: Any
    col: Any
    edge: Optional[Any] = None
    batch: Optional[Any] = None
    num_sampled_nodes: Optional[List[int]] = None
    num_sampled_edges: Optional[List[int]] = None
    orig_row: Optional[Any] = None
    orig_col: Optional[Any] = None
    metadata: Optional[Any] = None


@dataclass
class HeteroSamplerOutput:
    node: Dict[str, Any]
    row: Dict[Tuple[str, str, str], Any]
    col: Dict[Tuple[str, str, str], Any]
    edge: Dict[Tuple[str, str, str], Optional[Any]]
    batch: Optional[Dict[str, Any]] = None
    num_sampled_nodes: Optional[Dict[str, List[int]]] = None
    num_sampled_edges: Optional[Dict[Tuple[str, str, str], List[int]]] = None
    orig_row: Optional[Dict[Tuple[str, str, str], Any]] = None
    orig_col: Optional[Dict[Tuple[str, str, str], Any]] = None
    metadata: Optional[Any] = None


class NodeLoader(BaseDataLoader, AffinityMixin, MultithreadingMixin, LogMemoryMixin):
    r"""A data loader that performs mini-batch sampling from node information."""
    def __init__(
        self,
        data: Union[Data, HeteroData],
        node_sampler: Any,
        input_nodes: Any = None,
        input_time: Optional[Any] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional[Any] = None,
        input_id: Optional[Any] = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        self.data = data
        self.node_sampler = node_sampler
        self.input_nodes = input_nodes
        self.input_time = input_time
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls
        self.input_id = input_id

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        input_type, input_nodes, input_id = get_input_nodes(data, input_nodes, input_id)

        self.input_data = NodeSamplerInput(
            input_id=input_id,
            node=input_nodes,
            time=input_time,
            input_type=input_type,
        )

        num_inputs = input_nodes.size(0) if hasattr(input_nodes, 'size') else len(input_nodes)
        iterator = range(num_inputs)

        if torch is not None:
            super().__init__(iterator, collate_fn=self.collate_fn, **kwargs)
        else:
            self.dataset = iterator
            self.collate_fn = self.collate_fn

    def __call__(self, index: Any) -> Union[Data, HeteroData]:
        out = self.collate_fn(index)
        if not self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def collate_fn(self, index: Any) -> Any:
        input_data = self.input_data[index]
        out = self.node_sampler.sample_from_nodes(input_data)
        if self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def filter_fn(self, out: Any) -> Union[Data, HeteroData]:
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)

        if isinstance(out, SamplerOutput):
            perm = getattr(self.node_sampler, 'edge_permutation', None)
            data = filter_data(self.data, out.node, out.row, out.col, out.edge, perm)

            data.n_id = out.node
            if out.edge is not None:
                data.e_id = out.edge
            data.batch = out.batch
            data.num_sampled_nodes = out.num_sampled_nodes
            data.num_sampled_edges = out.num_sampled_edges

            meta = out.metadata or (out.node, None)
            data.input_id = meta[0]
            data.batch_size = meta[0].size(0) if hasattr(meta[0], 'size') else len(meta[0])

        elif isinstance(out, HeteroSamplerOutput):
            perm = getattr(self.node_sampler, 'edge_permutation', None)
            data = filter_hetero_data(self.data, out.node, out.row, out.col, out.edge, perm)

            for key, node in out.node.items():
                data[key].n_id = node

            for key, edge in (out.edge or {}).items():
                if edge is not None:
                    data[key].e_id = edge

            if out.batch is not None:
                data.set_value_dict('batch', out.batch)
            if out.num_sampled_nodes is not None:
                data.set_value_dict('num_sampled_nodes', out.num_sampled_nodes)
            if out.num_sampled_edges is not None:
                data.set_value_dict('num_sampled_edges', out.num_sampled_edges)

            input_type = self.input_data.input_type
            meta = out.metadata or (out.node.get(input_type, None), None)
            data[input_type].input_id = meta[0]
            data[input_type].batch_size = meta[0].size(0) if hasattr(meta[0], 'size') else len(meta[0])

        else:
            data = out

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

