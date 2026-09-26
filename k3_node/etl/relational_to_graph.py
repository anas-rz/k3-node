"""Relational (multi-table) to Heterogeneous Graph ETL converter."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from k3_node.data import HeteroData, Data
from k3_node.etl.encoders import TabularEncoder, _get_column_names, _get_column_values


NodeType = str
EdgeType = Tuple[str, str, str]


class RelationalToGraph:
    r"""ETL pipeline converting multi-table relational databases / DataFrames into
    a heterogeneous graph :class:`k3_node.data.HeteroData` object.

    Args:
        id_cols (dict): Mapping from :obj:`NodeType` to the primary key column name.
            (e.g., ``{"user": "user_id", "movie": "movie_id"}``).
        edge_cols (dict): Mapping from :obj:`EdgeType` to the tuple of foreign key column names
            ``(source_id_col, target_id_col)``.
            (e.g., ``{("user", "rates", "movie"): ("user_id", "movie_id")}``).
        feature_cols (dict, optional): Mapping from :obj:`NodeType` to a list of feature columns.
            If omitted, all columns except the primary key are encoded.
        edge_attr_cols (dict, optional): Mapping from :obj:`EdgeType` to a list of edge attribute columns.
        node_target_cols (dict, optional): Mapping from :obj:`NodeType` to the target label column.
        edge_target_cols (dict, optional): Mapping from :obj:`EdgeType` to the target edge label column.
    """

    def __init__(
        self,
        id_cols: Dict[NodeType, str],
        edge_cols: Dict[EdgeType, Tuple[str, str]],
        feature_cols: Optional[Dict[NodeType, List[str]]] = None,
        edge_attr_cols: Optional[Dict[EdgeType, List[str]]] = None,
        node_target_cols: Optional[Dict[NodeType, str]] = None,
        edge_target_cols: Optional[Dict[EdgeType, str]] = None,
    ):
        self.id_cols = id_cols
        self.edge_cols = edge_cols
        self.feature_cols = feature_cols or {}
        self.edge_attr_cols = edge_attr_cols or {}
        self.node_target_cols = node_target_cols or {}
        self.edge_target_cols = edge_target_cols or {}

        self.node_encoders_: Dict[NodeType, TabularEncoder] = {}
        self.edge_encoders_: Dict[EdgeType, TabularEncoder] = {}
        self.id_maps_: Dict[NodeType, Dict[Any, int]] = {}
        self.inverse_id_maps_: Dict[NodeType, Dict[int, Any]] = {}

    def fit(self, nodes: Dict[NodeType, Any], edges: Optional[Dict[EdgeType, Any]] = None):
        r"""Fits encoders and builds entity ID mappings across all tables."""
        # 1. Map node IDs and fit node feature encoders
        for node_type, table in nodes.items():
            id_col = self.id_cols[node_type]
            raw_ids = _get_column_values(table, id_col)

            # Unique contiguous ID mapping
            unique_ids = []
            seen = set()
            for rid in raw_ids:
                if rid not in seen:
                    seen.add(rid)
                    unique_ids.append(rid)

            id_map = {rid: i for i, rid in enumerate(unique_ids)}
            inv_map = {i: rid for i, rid in enumerate(unique_ids)}
            self.id_maps_[node_type] = id_map
            self.inverse_id_maps_[node_type] = inv_map

            # Feature columns
            all_cols = _get_column_names(table)
            target_col = self.node_target_cols.get(node_type)
            ignore_cols = {id_col}
            if target_col:
                ignore_cols.add(target_col)

            if node_type in self.feature_cols:
                feat_cols = [c for c in self.feature_cols[node_type] if c in all_cols and c not in ignore_cols]
            else:
                feat_cols = [c for c in all_cols if c not in ignore_cols]

            encoder = TabularEncoder()
            if feat_cols:
                encoder.fit(table, columns=feat_cols)
            self.node_encoders_[node_type] = encoder

        # 2. Fit edge encoders if edge attributes are specified
        if edges:
            for edge_type, table in edges.items():
                if edge_type in self.edge_attr_cols:
                    attr_cols = self.edge_attr_cols[edge_type]
                    edge_enc = TabularEncoder()
                    edge_enc.fit(table, columns=attr_cols)
                    self.edge_encoders_[edge_type] = edge_enc

        return self

    def transform(
        self,
        nodes: Dict[NodeType, Any],
        edges: Optional[Dict[EdgeType, Any]] = None,
    ) -> HeteroData:
        r"""Constructs a :class:`k3_node.data.HeteroData` instance from relational tables."""
        hetero_data = HeteroData()

        # 1. Process node tables
        for node_type, table in nodes.items():
            encoder = self.node_encoders_[node_type]
            if encoder.column_order_:
                x = encoder.transform(table)
                hetero_data[node_type].x = x
            else:
                num_nodes = len(self.id_maps_[node_type])
                hetero_data[node_type].num_nodes = num_nodes

            # Target labels y
            if node_type in self.node_target_cols:
                target_col = self.node_target_cols[node_type]
                raw_y = _get_column_values(table, target_col)
                if all(isinstance(v, (int, np.integer)) for v in raw_y if v is not None):
                    hetero_data[node_type].y = np.array(raw_y, dtype=np.int64)
                elif all(isinstance(v, (float, int, np.floating, np.integer)) for v in raw_y if v is not None):
                    hetero_data[node_type].y = np.array(raw_y, dtype=np.float32)
                else:
                    unique_c = sorted(list(set(raw_y)))
                    mapping = {c: i for i, c in enumerate(unique_c)}
                    hetero_data[node_type].y = np.array([mapping[c] for c in raw_y], dtype=np.int64)

        # 2. Process edge tables
        if edges:
            for edge_type, table in edges.items():
                src_type, rel_name, dst_type = edge_type
                src_col, dst_col = self.edge_cols[edge_type]

                raw_srcs = _get_column_values(table, src_col)
                raw_dsts = _get_column_values(table, dst_col)

                src_map = self.id_maps_[src_type]
                dst_map = self.id_maps_[dst_type]

                valid_src = []
                valid_dst = []
                valid_indices = []

                for row_idx, (s, d) in enumerate(zip(raw_srcs, raw_dsts)):
                    if s in src_map and d in dst_map:
                        valid_src.append(src_map[s])
                        valid_dst.append(dst_map[d])
                        valid_indices.append(row_idx)

                edge_index = np.array([valid_src, valid_dst], dtype=np.int64)
                hetero_data[edge_type].edge_index = edge_index

                # Edge attributes
                if edge_type in self.edge_encoders_:
                    edge_enc = self.edge_encoders_[edge_type]
                    full_attrs = edge_enc.transform(table)
                    hetero_data[edge_type].edge_attr = full_attrs[valid_indices]

                # Edge target y
                if edge_type in self.edge_target_cols:
                    t_col = self.edge_target_cols[edge_type]
                    raw_ey = _get_column_values(table, t_col)
                    filtered_ey = [raw_ey[idx] for idx in valid_indices]
                    hetero_data[edge_type].edge_label = np.array(filtered_ey, dtype=np.float32)

        # Attach metadata
        hetero_data.id_maps = self.id_maps_
        hetero_data.inverse_id_maps = self.inverse_id_maps_
        return hetero_data

    def fit_transform(
        self,
        nodes: Dict[NodeType, Any],
        edges: Optional[Dict[EdgeType, Any]] = None,
    ) -> HeteroData:
        r"""Fits encoders and builds the heterogeneous graph in a single call."""
        return self.fit(nodes, edges).transform(nodes, edges)


def relational_to_graph(
    nodes: Dict[NodeType, Any],
    edges: Optional[Dict[EdgeType, Any]] = None,
    id_cols: Optional[Dict[NodeType, str]] = None,
    edge_cols: Optional[Dict[EdgeType, Tuple[str, str]]] = None,
    **kwargs,
) -> HeteroData:
    r"""Functional shortcut to convert relational tables into a :class:`k3_node.data.HeteroData` graph."""
    if id_cols is None:
        raise ValueError("Must provide 'id_cols' mapping node types to their primary key columns.")
    if edges and edge_cols is None:
        raise ValueError("Must provide 'edge_cols' mapping edge types to (source_col, target_col).")

    etl = RelationalToGraph(
        id_cols=id_cols,
        edge_cols=edge_cols or {},
        **kwargs,
    )
    return etl.fit_transform(nodes, edges)
