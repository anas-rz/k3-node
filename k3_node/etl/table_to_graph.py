"""Table-to-Graph ETL converter for single tabular datasets."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np

from k3_node.data import Data
from k3_node.etl.encoders import TabularEncoder, _get_column_names, _get_column_values
from k3_node.etl.graph_builders import (
    KNNGraphBuilder,
    SimilarityGraphBuilder,
    SharedEntityGraphBuilder,
    SequentialGraphBuilder,
)


class TableToGraph:
    r"""ETL pipeline converting tabular datasets (DataFrames, CSVs, or dictionaries)
    into graph :class:`k3_node.data.Data` objects with node features, graph topology,
    labels, and split masks.

    Args:
        feature_cols: List of column names used as node features. If :obj:`None`, all
            columns except :obj:`target_col` and :obj:`id_col` are used.
        target_col (str, optional): Target column used to create ground-truth labels :obj:`y`.
        id_col (str, optional): Unique row identifier column (e.g. customer_id, product_id).
            Stored in :obj:`data.node_ids` and :obj:`data.id_to_index`.
        edge_strategy (str or callable): Strategy for constructing edges (``"knn"``,
            ``"similarity"``, ``"shared_entity"``, ``"sequential"``, or a custom callable).
            (default: ``"knn"``)
        edge_kwargs (dict, optional): Keyword arguments forwarded to the edge builder.
            (e.g., ``{"k": 5, "metric": "cosine"}`` for KNN).
        column_encoders (dict, optional): Explicit mapping from column names to encoder instances.
        train_ratio (float, optional): Fraction of nodes for training mask. (default: ``None``)
        val_ratio (float, optional): Fraction of nodes for validation mask. (default: ``None``)
        test_ratio (float, optional): Fraction of nodes for test mask. (default: ``None``)
        random_state (int, optional): Random seed for mask splits. (default: ``42``)
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        id_col: Optional[str] = None,
        edge_strategy: Union[str, Callable] = "knn",
        edge_kwargs: Optional[Dict[str, Any]] = None,
        column_encoders: Optional[Dict[str, Any]] = None,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
        random_state: int = 42,
    ):
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.id_col = id_col
        self.edge_strategy = edge_strategy
        self.edge_kwargs = edge_kwargs or {}
        self.column_encoders = column_encoders or {}
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        self.encoder_ = TabularEncoder(column_encoders=self.column_encoders)
        self.edge_builder_ = self._resolve_edge_builder()

    def _resolve_edge_builder(self) -> Callable:
        if callable(self.edge_strategy):
            return self.edge_strategy

        strategy = str(self.edge_strategy).lower()
        if strategy == "knn":
            return KNNGraphBuilder(**self.edge_kwargs)
        elif strategy in ("similarity", "sim"):
            return SimilarityGraphBuilder(**self.edge_kwargs)
        elif strategy in ("shared_entity", "shared", "bipartite"):
            return SharedEntityGraphBuilder(**self.edge_kwargs)
        elif strategy in ("sequential", "sequence", "temporal"):
            return SequentialGraphBuilder(**self.edge_kwargs)
        else:
            raise ValueError(
                f"Unknown edge_strategy '{self.edge_strategy}'. "
                f"Available: 'knn', 'similarity', 'shared_entity', 'sequential', or custom callable."
            )

    def fit(self, df_or_dict: Any):
        r"""Fits the tabular feature encoder on the input table."""
        all_cols = _get_column_names(df_or_dict)
        ignore_cols = set()
        if self.target_col:
            ignore_cols.add(self.target_col)
        if self.id_col:
            ignore_cols.add(self.id_col)

        if self.feature_cols is None:
            fit_cols = [c for c in all_cols if c not in ignore_cols]
        else:
            fit_cols = [c for c in self.feature_cols if c in all_cols and c not in ignore_cols]

        self.encoder_.fit(df_or_dict, columns=fit_cols)
        return self

    def transform(self, df_or_dict: Any) -> Data:
        r"""Encodes features, constructs graph edges, and builds a :class:`Data` object."""
        # 1. Node features
        x = self.encoder_.transform(df_or_dict)
        num_nodes = x.shape[0]

        # 2. Graph topology
        edge_index, edge_attr = self.edge_builder_(x=x, df_or_dict=df_or_dict)

        # 3. Target labels y
        y = None
        if self.target_col is not None:
            raw_y = _get_column_values(df_or_dict, self.target_col)
            # Check if categorical or numerical
            if all(isinstance(v, (int, np.integer)) for v in raw_y if v is not None):
                y = np.array(raw_y, dtype=np.int64)
            elif all(isinstance(v, (float, int, np.floating, np.integer)) for v in raw_y if v is not None):
                y = np.array(raw_y, dtype=np.float32)
            else:
                # String labels -> label encode to ints
                unique_classes = sorted(list(set(raw_y)))
                mapping = {c: i for i, c in enumerate(unique_classes)}
                y = np.array([mapping[c] for c in raw_y], dtype=np.int64)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # 4. Optional entity ID tracking
        if self.id_col is not None:
            raw_ids = _get_column_values(df_or_dict, self.id_col)
            data.node_ids = list(raw_ids)
            data.id_to_index = {raw_id: idx for idx, raw_id in enumerate(raw_ids)}
            data.index_to_id = {idx: raw_id for idx, raw_id in enumerate(raw_ids)}

        # 5. Split masks
        if self.train_ratio is not None:
            np.random.seed(self.random_state)
            indices = np.random.permutation(num_nodes)

            n_train = int(num_nodes * self.train_ratio)
            val_ratio = self.val_ratio or 0.0
            n_val = int(num_nodes * val_ratio)

            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

            train_mask = np.zeros(num_nodes, dtype=bool)
            train_mask[train_idx] = True
            data.train_mask = train_mask

            if val_ratio > 0:
                val_mask = np.zeros(num_nodes, dtype=bool)
                val_mask[val_idx] = True
                data.val_mask = val_mask

            if self.test_ratio is not None or len(test_idx) > 0:
                test_mask = np.zeros(num_nodes, dtype=bool)
                test_mask[test_idx] = True
                data.test_mask = test_mask

        return data

    def fit_transform(self, df_or_dict: Any) -> Data:
        r"""Fits encoders and transforms tabular data into a graph in a single call."""
        return self.fit(df_or_dict).transform(df_or_dict)

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        id_col: Optional[str] = None,
        edge_strategy: Union[str, Callable] = "knn",
        **kwargs,
    ) -> Data:
        r"""Convenience factory method directly converting a pandas DataFrame into a :class:`Data` object."""
        etl = cls(
            feature_cols=feature_cols,
            target_col=target_col,
            id_col=id_col,
            edge_strategy=edge_strategy,
            **kwargs,
        )
        return etl.fit_transform(df)

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        id_col: Optional[str] = None,
        edge_strategy: Union[str, Callable] = "knn",
        **kwargs,
    ) -> Data:
        r"""Convenience factory method directly loading a CSV file and converting it into a :class:`Data` object."""
        import csv

        # Parse CSV into dict of column lists
        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data_dict: Dict[str, List[Any]] = {field: [] for field in reader.fieldnames or []}
            for row in reader:
                for k, v in row.items():
                    # Attempt numeric cast
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
                    data_dict[k].append(v)

        return cls.from_dataframe(
            df=data_dict,
            feature_cols=feature_cols,
            target_col=target_col,
            id_col=id_col,
            edge_strategy=edge_strategy,
            **kwargs,
        )


# Alias
TabularToGraph = TableToGraph


def table_to_graph(
    df_or_dict: Any,
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    id_col: Optional[str] = None,
    edge_strategy: Union[str, Callable] = "knn",
    **kwargs,
) -> Data:
    r"""Functional shortcut to convert a table or dictionary into a :class:`k3_node.data.Data` object."""
    etl = TableToGraph(
        feature_cols=feature_cols,
        target_col=target_col,
        id_col=id_col,
        edge_strategy=edge_strategy,
        **kwargs,
    )
    return etl.fit_transform(df_or_dict)
