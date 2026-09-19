from typing import Any, Callable, Dict, List, Optional, Tuple, Union


from k3_node.data import HeteroData
from k3_node.loader.node_loader import HeteroSamplerOutput, NodeLoader, NodeSamplerInput
from k3_node.loader.sampler_utils import sample_neighbors_hetero


class InternalHGTSampler:
    r"""HGT balanced neighborhood sampling engine."""
    def __init__(
        self,
        data: HeteroData,
        num_samples: Union[List[int], Dict[str, List[int]]],
    ):
        self.data = data
        self.num_samples = num_samples
        self.edge_permutation = None

    def sample_from_nodes(self, input_data: NodeSamplerInput) -> HeteroSamplerOutput:
        edge_index_dict = {}
        for edge_type in self.data.edge_types:
            canonical = self.data._to_canonical(*edge_type) if hasattr(self.data, '_to_canonical') else edge_type
            edge_index_dict[canonical] = self.data[edge_type].edge_index

        node_type = input_data.input_type or self.data.node_types[0]
        seed_dict = {k: None for k in self.data.node_types}
        seed_dict[node_type] = input_data.node

        # Build num_neighbors dict for hetero sampling
        if isinstance(self.num_samples, dict):
            num_neighbors = {}
            for e in self.data.edge_types:
                dst = e[2]
                can = self.data._to_canonical(*e) if hasattr(self.data, '_to_canonical') else e
                num_neighbors[can] = self.num_samples.get(dst, [10])
        else:
            num_neighbors = self.num_samples

        node_dict, row_dict, col_dict, edge_dict, n_counts, e_counts = sample_neighbors_hetero(
            edge_index_dict=edge_index_dict,
            seed_nodes_dict=seed_dict,
            num_neighbors=num_neighbors,
            replace=True,
            subgraph_type='directional',
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


class HGTLoader(NodeLoader):
    r"""The Heterogeneous Graph Sampler from the "Heterogeneous Graph Transformer" paper.

    Args:
        data (HeteroData): The heterogeneous graph data object.
        num_samples (List[int] or Dict[str, List[int]]): The number of nodes to sample per iteration.
        input_nodes (str or Tuple[str, Tensor]): Seed node type and indices.
        **kwargs (optional): Additional arguments of :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        data: HeteroData,
        num_samples: Union[List[int], Dict[str, List[int]]],
        input_nodes: Union[str, Tuple[str, Optional[Any]]],
        is_sorted: bool = False,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        **kwargs,
    ):
        hgt_sampler = InternalHGTSampler(data, num_samples=num_samples)

        super().__init__(
            data=data,
            node_sampler=hgt_sampler,
            input_nodes=input_nodes,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )

