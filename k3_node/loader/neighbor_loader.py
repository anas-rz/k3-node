from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from k3_node.data import Data, HeteroData
from k3_node.loader.node_loader import HeteroSamplerOutput, NodeLoader, NodeSamplerInput, SamplerOutput
from k3_node.loader.sampler_utils import sample_neighbors_hetero, sample_neighbors_homo


class InternalNeighborSampler:
    r"""Pure Python/NumPy neighborhood sampling engine."""
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: Union[List[int], Dict[Tuple[str, str, str], List[int]]],
        replace: bool = False,
        subgraph_type: str = 'directional',
        disjoint: bool = False,
    ):
        self.data = data
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.subgraph_type = subgraph_type
        self.disjoint = disjoint
        self.edge_permutation = None

    def sample_from_nodes(self, input_data: NodeSamplerInput) -> Union[SamplerOutput, HeteroSamplerOutput]:
        if isinstance(self.data, Data):
            node, row, col, edge, n_counts, e_counts = sample_neighbors_homo(
                edge_index=self.data.edge_index,
                seed_nodes=input_data.node,
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
                metadata=(input_data.input_id, input_data.time),
            )
        elif isinstance(self.data, HeteroData):
            edge_index_dict = {}
            for edge_type in self.data.edge_types:
                canonical = self.data._to_canonical(*edge_type) if hasattr(self.data, '_to_canonical') else edge_type
                edge_index_dict[canonical] = self.data[edge_type].edge_index

            node_type = input_data.input_type or self.data.node_types[0]
            seed_dict = {k: None for k in self.data.node_types}
            seed_dict[node_type] = input_data.node

            # Normalize num_neighbors for hetero
            if isinstance(self.num_neighbors, dict):
                norm_num_neighbors = {}
                for k, v in self.num_neighbors.items():
                    can = self.data._to_canonical(*k) if hasattr(self.data, '_to_canonical') else k
                    norm_num_neighbors[can] = v
            else:
                norm_num_neighbors = self.num_neighbors

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
                metadata=(input_data.input_id, input_data.time),
            )

        raise TypeError(f"Invalid data type for sampling: {type(self.data)}")


class NeighborLoader(NodeLoader):
    r"""A data loader that performs neighbor sampling as introduced in
    "Inductive Representation Learning on Large Graphs".

    Args:
        data (Data or HeteroData): The graph data object.
        num_neighbors (List[int] or Dict[EdgeType, List[int]]): Number of neighbors to sample per iteration.
        input_nodes (Tensor or str or Tuple[str, Tensor], optional): Seed nodes. (default: :obj:`None`)
        replace (bool, optional): Sample with replacement. (default: :obj:`False`)
        subgraph_type (str, optional): :obj:`"directional"`, :obj:`"bidirectional"`, or :obj:`"induced"`.
            (default: :obj:`"directional"`)
        disjoint (bool, optional): If :obj:`True`, creates disjoint subgraphs per seed node. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        data: Union[Data, HeteroData],
        num_neighbors: Union[List[int], Dict[Tuple[str, str, str], List[int]]],
        input_nodes: Any = None,
        input_time: Optional[Any] = None,
        replace: bool = False,
        subgraph_type: str = 'directional',
        disjoint: bool = False,
        temporal_strategy: str = 'uniform',
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

        if neighbor_sampler is None:
            neighbor_sampler = InternalNeighborSampler(
                data,
                num_neighbors=num_neighbors,
                replace=replace,
                subgraph_type=str(getattr(subgraph_type, 'value', subgraph_type)),
                disjoint=disjoint,
            )

        super().__init__(
            data=data,
            node_sampler=neighbor_sampler,
            input_nodes=input_nodes,
            input_time=input_time,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )

