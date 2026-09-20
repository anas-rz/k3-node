from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor
except ImportError:
    torch = None
    Tensor = type(None)

from k3_node.data import Data, HeteroData
from k3_node.loader.base import BaseDataLoader, DataLoaderIterator
from k3_node.loader.mixin import AffinityMixin, LogMemoryMixin, MultithreadingMixin
from k3_node.loader.node_loader import HeteroSamplerOutput, SamplerOutput
from k3_node.loader.utils import (
    filter_data,
    filter_hetero_data,
    get_edge_label_index,
    infer_filter_per_worker,
)


@dataclass
class EdgeSamplerInput:
    input_id: Optional[Any]
    row: Any
    col: Any
    label: Optional[Any] = None
    time: Optional[Any] = None
    input_type: Optional[Tuple[str, str, str]] = None

    def __getitem__(self, index: Any) -> 'EdgeSamplerInput':
        if torch is not None and not isinstance(index, Tensor):
            index = torch.as_tensor(index, dtype=torch.long)
        return EdgeSamplerInput(
            input_id=self.input_id[index] if self.input_id is not None else index,
            row=self.row[index],
            col=self.col[index],
            label=self.label[index] if self.label is not None else None,
            time=self.time[index] if self.time is not None else None,
            input_type=self.input_type,
        )


class LinkLoader(BaseDataLoader, AffinityMixin, MultithreadingMixin, LogMemoryMixin):
    r"""A data loader that performs mini-batch sampling from link information."""
    def __init__(
        self,
        data: Union[Data, HeteroData],
        link_sampler: Any,
        edge_label_index: Any = None,
        edge_label: Optional[Any] = None,
        edge_label_time: Optional[Any] = None,
        neg_sampling: Optional[Any] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional[Any] = None,
        input_id: Optional[Any] = None,
        **kwargs,
    ):
        if filter_per_worker is None:
            filter_per_worker = infer_filter_per_worker(data)

        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)

        input_type, edge_label_index = get_edge_label_index(data, edge_label_index)

        self.data = data
        self.link_sampler = link_sampler
        self.neg_sampling = neg_sampling
        self.neg_sampling_ratio = neg_sampling_ratio
        self.transform = transform
        self.transform_sampler_output = transform_sampler_output
        self.filter_per_worker = filter_per_worker
        self.custom_cls = custom_cls

        if torch is not None and isinstance(edge_label_index, Tensor):
            row = edge_label_index[0]
            col = edge_label_index[1]
            num_edges = edge_label_index.size(1)
        else:
            np_edges = np.asarray(edge_label_index)
            row = np_edges[0]
            col = np_edges[1]
            num_edges = np_edges.shape[1]

        self.input_data = EdgeSamplerInput(
            input_id=input_id,
            row=row,
            col=col,
            label=edge_label,
            time=edge_label_time,
            input_type=input_type,
        )

        iterator = range(num_edges)

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
        out = self.link_sampler.sample_from_edges(input_data)
        if self.filter_per_worker:
            out = self.filter_fn(out)
        return out

    def filter_fn(self, out: Any) -> Union[Data, HeteroData]:
        if self.transform_sampler_output:
            out = self.transform_sampler_output(out)

        if isinstance(out, SamplerOutput):
            perm = getattr(self.link_sampler, 'edge_permutation', None)
            data = filter_data(self.data, out.node, out.row, out.col, out.edge, perm)

            data.n_id = out.node
            if out.edge is not None:
                data.e_id = out.edge
            data.batch = out.batch
            data.num_sampled_nodes = out.num_sampled_nodes
            data.num_sampled_edges = out.num_sampled_edges

            meta = out.metadata or (None, None, None)
            data.input_id = meta[0]
            data.edge_label_index = meta[1]
            data.edge_label = meta[2]

        elif isinstance(out, HeteroSamplerOutput):
            perm = getattr(self.link_sampler, 'edge_permutation', None)
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
            meta = out.metadata or (None, None, None)
            if input_type is not None:
                data[input_type].input_id = meta[0]
                data[input_type].edge_label_index = meta[1]
                data[input_type].edge_label = meta[2]
        else:
            data = out

        return data if self.transform is None else self.transform(data)

    def _get_iterator(self) -> Iterator:
        if self.filter_per_worker:
            return super()._get_iterator()
        return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'

