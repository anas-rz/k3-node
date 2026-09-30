"""Subgraph extraction utilities for GraphRAG and Knowledge Graph LLM integration."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from keras import ops

from k3_node.data import Data


@dataclass
class SubgraphResult:
    """Structured result of a subgraph extraction around retrieved entities.

    Attributes:
        edge_index: Tensor of shape `(2, num_edges)` with subgraph edges.
        edge_type: Optional tensor of shape `(num_edges,)` with relation types.
        edge_attr: Optional tensor of shape `(num_edges, edge_dim)` with edge features.
        x: Optional tensor of shape `(num_nodes, in_channels)` with node features.
        nodes: 1D numpy array of original node indices in the full graph.
        center_nodes: 1D numpy array of center entity indices in the relabeled subgraph.
        mapping: Dictionary mapping original node index to subgraph node index.
        edge_mask: 1D boolean array indicating edges kept from original graph.
        num_nodes: Number of nodes in the extracted subgraph.
        num_edges: Number of edges in the extracted subgraph.
    """

    edge_index: any
    edge_type: Optional[any] = None
    edge_attr: Optional[any] = None
    x: Optional[any] = None
    nodes: Optional[np.ndarray] = None
    center_nodes: Optional[np.ndarray] = None
    mapping: Optional[Dict[int, int]] = None
    edge_mask: Optional[np.ndarray] = None
    num_nodes: int = 0
    num_edges: int = 0

    def to_data(self) -> Data:
        """Convert extracted subgraph into a K3-Node Data object."""
        return Data(
            x=self.x,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_attr=self.edge_attr,
            center_nodes=ops.convert_to_tensor(self.center_nodes, dtype="int32")
            if self.center_nodes is not None
            else None,
            original_nodes=ops.convert_to_tensor(self.nodes, dtype="int32")
            if self.nodes is not None
            else None,
        )


def extract_subgraph(
    entities: Union[int, Sequence[int], np.ndarray, any],
    edge_index: Union[np.ndarray, any],
    edge_type: Optional[Union[np.ndarray, any]] = None,
    edge_attr: Optional[Union[np.ndarray, any]] = None,
    x: Optional[Union[np.ndarray, any]] = None,
    num_hops: int = 2,
    max_nodes_per_hop: Optional[int] = None,
    directed: bool = False,
    relabel_nodes: bool = True,
    num_nodes: Optional[int] = None,
) -> SubgraphResult:
    """Extract multi-hop enclosing or ego-subgraphs around retrieved entities.

    Given a knowledge graph or relational graph, this function expands `num_hops`
    around seed `entities`, keeping all induced edges, relation types, and node/edge
    attributes.

    Args:
        entities: Single node index or list/array of seed entity indices.
        edge_index: Graph connectivity tensor of shape `(2, num_edges)`.
        edge_type: Optional 1D relation type tensor of shape `(num_edges,)`.
        edge_attr: Optional edge attribute tensor of shape `(num_edges, edge_dim)`.
        x: Optional node feature tensor of shape `(total_nodes, in_channels)`.
        num_hops: Number of hops to expand around retrieved entities. (default: 2)
        max_nodes_per_hop: Optional maximum number of neighboring nodes to keep
            per hop (useful to restrict explosion on hub entities).
        directed: If True, only follows outgoing edges. If False, follows edges
            in both directions (standard for KG context expansion). (default: False)
        relabel_nodes: If True, relabels subgraph node IDs to `0..num_subgraph_nodes-1`.
            (default: True)
        num_nodes: Optional total number of nodes in graph. Inferred if not given.

    Returns:
        `SubgraphResult` containing relabeled edge_index, edge_type, features,
        and entity mappings.
    """
    edge_index_np = np.asarray(ops.convert_to_numpy(edge_index)).astype(np.int64)
    if num_nodes is None:
        num_nodes = int(edge_index_np.max()) + 1 if edge_index_np.size > 0 else 0

    entities_arr = np.atleast_1d(np.asarray(ops.convert_to_numpy(entities))).astype(np.int64)
    if entities_arr.size == 0:
        empty_ei = ops.convert_to_tensor(np.zeros((2, 0), dtype=np.int64), dtype=edge_index.dtype)
        return SubgraphResult(
            edge_index=empty_ei,
            nodes=np.array([], dtype=np.int64),
            center_nodes=np.array([], dtype=np.int64),
            mapping={},
            num_nodes=0,
            num_edges=0,
        )

    row, col = edge_index_np[0], edge_index_np[1]
    subsets = [entities_arr]
    visited = set(entities_arr.tolist())

    for _ in range(num_hops):
        current_frontier = subsets[-1]
        if len(current_frontier) == 0:
            break

        mask_frontier = np.zeros(num_nodes, dtype=bool)
        mask_frontier[current_frontier] = True

        # If undirected (or GraphRAG bidirectional expansion), collect neighbors in both directions
        if not directed:
            neighbors_out = col[mask_frontier[row]]
            neighbors_in = row[mask_frontier[col]]
            new_neighbors = np.concatenate([neighbors_out, neighbors_in])
        else:
            new_neighbors = col[mask_frontier[row]]

        if new_neighbors.size > 0:
            unique_new = np.unique(new_neighbors)
            unseen = [n for n in unique_new if n not in visited]
            if max_nodes_per_hop is not None and len(unseen) > max_nodes_per_hop:
                unseen = unseen[:max_nodes_per_hop]
            visited.update(unseen)
            subsets.append(np.array(unseen, dtype=np.int64))
        else:
            break

    subgraph_nodes = np.unique(np.concatenate([s for s in subsets if len(s) > 0]))

    # Induced edge mask: both endpoints must be in subgraph_nodes
    node_mask = np.zeros(num_nodes, dtype=bool)
    node_mask[subgraph_nodes] = True
    edge_mask = node_mask[row] & node_mask[col]

    sub_edge_index = edge_index_np[:, edge_mask]

    mapping = {int(orig): int(new_idx) for new_idx, orig in enumerate(subgraph_nodes)}
    center_indices = np.array([mapping[int(e)] for e in entities_arr if int(e) in mapping], dtype=np.int64)

    if relabel_nodes:
        new_id_table = np.full(num_nodes, -1, dtype=np.int64)
        new_id_table[subgraph_nodes] = np.arange(len(subgraph_nodes), dtype=np.int64)
        sub_edge_index = new_id_table[sub_edge_index]

    # Preserve tensors in their original backend / dtype
    sub_edge_index_tensor = ops.convert_to_tensor(sub_edge_index, dtype=edge_index.dtype)

    sub_edge_type_tensor = None
    if edge_type is not None:
        et_np = np.asarray(ops.convert_to_numpy(edge_type))[edge_mask]
        sub_edge_type_tensor = ops.convert_to_tensor(et_np, dtype=edge_type.dtype)

    sub_edge_attr_tensor = None
    if edge_attr is not None:
        ea_np = np.asarray(ops.convert_to_numpy(edge_attr))[edge_mask]
        sub_edge_attr_tensor = ops.convert_to_tensor(ea_np, dtype=edge_attr.dtype)

    sub_x_tensor = None
    if x is not None:
        x_np = np.asarray(ops.convert_to_numpy(x))[subgraph_nodes]
        sub_x_tensor = ops.convert_to_tensor(x_np, dtype=x.dtype)

    return SubgraphResult(
        edge_index=sub_edge_index_tensor,
        edge_type=sub_edge_type_tensor,
        edge_attr=sub_edge_attr_tensor,
        x=sub_x_tensor,
        nodes=subgraph_nodes,
        center_nodes=center_indices,
        mapping=mapping,
        edge_mask=edge_mask,
        num_nodes=len(subgraph_nodes),
        num_edges=int(sub_edge_index.shape[1]),
    )


class KGEntityRetriever:
    """Knowledge Graph Entity and Subgraph Retriever for GraphRAG.

    Maintains entity name dictionaries and relation mappings, extracts seed entities
    from text queries, and retrieves enclosing multi-hop subgraphs.

    Args:
        entity_to_id: Dictionary mapping entity strings to node integer IDs.
        relation_to_id: Dictionary mapping relation strings to edge_type integer IDs.
        edge_index: Graph connectivity tensor of shape `(2, num_edges)`.
        edge_type: Optional relation type tensor of shape `(num_edges,)`.
        edge_attr: Optional edge feature tensor.
        x: Optional node feature tensor.
    """

    def __init__(
        self,
        entity_to_id: Dict[str, int],
        relation_to_id: Dict[str, int],
        edge_index: Union[np.ndarray, any],
        edge_type: Optional[Union[np.ndarray, any]] = None,
        edge_attr: Optional[Union[np.ndarray, any]] = None,
        x: Optional[Union[np.ndarray, any]] = None,
    ):
        self.entity_to_id = entity_to_id
        self.relation_to_id = relation_to_id
        self.id_to_entity = {v: k for k, v in entity_to_id.items()}
        self.id_to_relation = {v: k for k, v in relation_to_id.items()}

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_attr = edge_attr
        self.x = x

    def get_entity_id(self, name: str) -> Optional[int]:
        """Look up entity ID by exact name (case-insensitive fallback)."""
        if name in self.entity_to_id:
            return self.entity_to_id[name]
        # Case-insensitive fallback
        lower_map = {k.lower(): v for k, v in self.entity_to_id.items()}
        return lower_map.get(name.lower(), None)

    def find_entities_in_text(self, text: str) -> List[str]:
        """Find matching known entity names mentioned in a query text."""
        text_lower = text.lower()
        matched = []
        # Sort by length descending to match longest phrases first
        for name in sorted(self.entity_to_id.keys(), key=lambda s: len(s), reverse=True):
            if name.lower() in text_lower:
                matched.append(name)
        return matched

    def retrieve_subgraph(
        self,
        entities: Union[Sequence[Union[str, int]], str, int],
        num_hops: int = 2,
        max_nodes_per_hop: Optional[int] = None,
        directed: bool = False,
    ) -> SubgraphResult:
        """Extract multi-hop subgraph around the specified entity names or IDs."""
        if isinstance(entities, (str, int)):
            entities = [entities]

        entity_ids = []
        for e in entities:
            if isinstance(e, str):
                eid = self.get_entity_id(e)
                if eid is not None:
                    entity_ids.append(eid)
            else:
                entity_ids.append(int(e))

        return extract_subgraph(
            entities=entity_ids,
            edge_index=self.edge_index,
            edge_type=self.edge_type,
            edge_attr=self.edge_attr,
            x=self.x,
            num_hops=num_hops,
            max_nodes_per_hop=max_nodes_per_hop,
            directed=directed,
            relabel_nodes=True,
        )
