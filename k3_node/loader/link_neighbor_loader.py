from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch import Tensor
except ImportError:
    torch = None
    Tensor = type(None)

from k3_node.data import Data, HeteroData
from k3_node.loader.link_loader import EdgeSamplerInput, LinkLoader
from k3_node.loader.node_loader import HeteroSamplerOutput, SamplerOutput
from k3_node.loader.sampler_utils import sample_neighbors_hetero, sample_neighbors_homo


class InternalLinkNeighborSampler:
    r"""Pure Python/NumPy link neighborhood sampler."""
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: Union[List[int], Dict[Tuple[str, str, str], List[int]]],
        replace: bool = False,
        subgraph_type: str = 'directional',
        disjoint: bool = False,
        neg_sampling_ratio: float = 0.0,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.subgraph_type = subgraph_type
        self.disjoint = disjoint
        self.neg_sampling_ratio = neg_sampling_ratio
        self.edge_permutation = None

    def sample_from_edges(self, input_data: EdgeSamplerInput) -> Union[SamplerOutput, HeteroSamplerOutput]:
        is_torch = torch is not None and isinstance(input_data.row, Tensor)

        pos_row = input_data.row
        pos_col = input_data.col
        pos_len = pos_row.size(0) if is_torch else len(pos_row)

        if self.neg_sampling_ratio > 0:
            num_neg = round(self.neg_sampling_ratio * pos_len)
            num_total_nodes = self.data.num_nodes
            if is_torch:
                neg_col = torch.randint(0, num_total_nodes, (num_neg,), dtype=torch.long, device=pos_row.device)
                neg_row = pos_row[torch.randint(0, pos_len, (num_neg,), dtype=torch.long, device=pos_row.device)]
                total_row = torch.cat([pos_row, neg_row], dim=0)
                total_col = torch.cat([pos_col, neg_col], dim=0)
                edge_label = torch.cat([torch.ones(pos_len, dtype=torch.float, device=pos_row.device),
                                        torch.zeros(num_neg, dtype=torch.float, device=pos_row.device)], dim=0)
            else:
                neg_col = np.random.randint(0, num_total_nodes, size=(num_neg,), dtype=np.int64)
                neg_row = np.random.choice(pos_row, size=(num_neg,), replace=True)
                total_row = np.concatenate([pos_row, neg_row], axis=0)
                total_col = np.concatenate([pos_col, neg_col], axis=0)
                edge_label = np.concatenate([np.ones(pos_len, dtype=np.float32), np.zeros(num_neg, dtype=np.float32)], axis=0)
        else:
            total_row = pos_row
            total_col = pos_col
            edge_label = input_data.label

        if is_torch:
            seed_nodes = torch.cat([total_row, total_col], dim=0).unique()
            edge_label_index = torch.stack([total_row, total_col], dim=0)
        else:
            seed_nodes = np.unique(np.concatenate([total_row, total_col], axis=0))
            edge_label_index = np.stack([total_row, total_col], axis=0)

        if isinstance(self.data, Data):
            node, row, col, edge, n_counts, e_counts = sample_neighbors_homo(
                edge_index=self.data.edge_index,
                seed_nodes=seed_nodes,
                num_neighbors=self.num_neighbors,
                num_nodes=self.data.num_nodes,
                replace=self.replace,
                subgraph_type=self.subgraph_type,
                disjoint=self.disjoint,
            )
            return SamplerOutput(
                node=node,
                row=row,
                col=col,
                edge=edge,
                num_sampled_nodes=n_counts,
                num_sampled_edges=e_counts,
                metadata=(input_data.input_id, edge_label_index, edge_label),
            )
        elif isinstance(self.data, HeteroData):
            edge_index_dict = {k: self.data[k].edge_index for k in self.data.edge_types}
            input_type = input_data.input_type or self.data.edge_types[0]
            src_type, _, dst_type = input_type

            seed_dict = {k: None for k in self.data.node_types}
            if src_type == dst_type:
                seed_dict[src_type] = seed_nodes
            else:
                seed_dict[src_type] = total_row
                seed_dict[dst_type] = total_col

            norm_num_neighbors = self.num_neighbors
            if isinstance(norm_num_neighbors, dict):
                norm_num_neighbors = {self.data._to_canonical(*k): v for k, v in norm_num_neighbors.items()}

            node_dict, row_dict, col_dict, edge_dict, n_counts, e_counts = sample_neighbors_hetero(
                edge_index_dict=edge_index_dict,
                seed_nodes_dict=seed_dict,
                num_neighbors=norm_num_neighbors,
                replace=self.replace,
                subgraph_type=self.subgraph_type,
            )
            return HeteroSamplerOutput(
                node=node_dict,
                row=row_dict,
                col=col_dict,
                edge=edge_dict,
                num_sampled_nodes=n_counts,
                num_sampled_edges=e_counts,
                metadata=(input_data.input_id, edge_label_index, edge_label),
            )

        raise TypeError(f"Invalid data type: {type(self.data)}")


class LinkNeighborLoader(LinkLoader):
    r"""A link-based data loader derived as an extension of NeighborLoader."""
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: Union[List[int], Dict[Tuple[str, str, str], List[int]]],
        edge_label_index: Any = None,
        edge_label: Optional[Any] = None,
        edge_label_time: Optional[Any] = None,
        replace: bool = False,
        subgraph_type: str = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
        neg_sampling: Optional[Any] = None,
        neg_sampling_ratio: Optional[Union[int, float]] = None,
        time_attr: Optional[str] = None,
        weight_attr: Optional[str] = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        is_sorted: bool = False,
        filter_per_worker: Optional[bool] = None,
        neighbor_sampler: Optional[Any] = None,
        directed: bool = True,
        **kwargs,
    ):
        if not directed:
            subgraph_type = 'induced'

        ratio = 0.0
        if neg_sampling_ratio is not None:
            ratio = float(neg_sampling_ratio)
        elif neg_sampling is not None:
            ratio = float(getattr(neg_sampling, 'amount', 1.0)) if hasattr(neg_sampling, 'amount') else 1.0

        if neighbor_sampler is None:
            neighbor_sampler = InternalLinkNeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=str(getattr(subgraph_type, 'value', subgraph_type)),
                disjoint=disjoint,
                neg_sampling_ratio=ratio,
            )

        super().__init__(
            data=data,
            link_sampler=neighbor_sampler,
            edge_label_index=edge_label_index,
            edge_label=edge_label,
            edge_label_time=edge_label_time,
            neg_sampling=neg_sampling,
            neg_sampling_ratio=neg_sampling_ratio,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )

