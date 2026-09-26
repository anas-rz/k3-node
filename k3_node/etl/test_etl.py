"""Unit tests for K3-Node Tabular-to-Graph ETL pipelines."""

import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from keras import ops

import k3_node
from k3_node.data import Data, HeteroData
from k3_node.etl import (
    NumericalEncoder,
    CategoricalEncoder,
    TabularEncoder,
    KNNGraphBuilder,
    SimilarityGraphBuilder,
    SharedEntityGraphBuilder,
    SequentialGraphBuilder,
    TableToGraph,
    TabularToGraph,
    table_to_graph,
    RelationalToGraph,
    relational_to_graph,
)
from k3_node.tasks import NodeClassifier


# ==============================================================================
# 1. Encoders Unit Tests
# ==============================================================================

def test_numerical_encoder():
    raw = [1.0, 2.0, 3.0, np.nan, 5.0]

    # Standard scaling
    enc_std = NumericalEncoder(strategy="standard", impute_strategy="mean")
    res_std = enc_std.fit_transform(raw)
    assert res_std.shape == (5, 1)
    assert not np.isnan(res_std).any()

    # MinMax scaling
    enc_mm = NumericalEncoder(strategy="minmax", impute_strategy="zero")
    res_mm = enc_mm.fit_transform(raw)
    assert res_mm.shape == (5, 1)
    assert res_mm.min() >= 0.0 and res_mm.max() <= 1.0

    # Log1p
    enc_log = NumericalEncoder(strategy="log1p")
    res_log = enc_log.fit_transform([0.0, 1.0, 10.0])
    assert np.allclose(res_log[0, 0], 0.0)


def test_categorical_encoder():
    raw = ["apple", "banana", "apple", None, "orange"]

    # One-hot
    enc_oh = CategoricalEncoder(strategy="onehot", handle_unknown="ignore")
    res_oh = enc_oh.fit_transform(raw)
    assert res_oh.shape[0] == 5
    assert res_oh.shape[1] >= 3

    # Unknown category handling
    transformed = enc_oh.transform(["grape", "banana"])
    assert transformed.shape == (2, res_oh.shape[1])
    assert transformed[0].sum() == 0.0  # unknown ignored
    assert transformed[1].sum() == 1.0  # banana matched

    # Ordinal
    enc_ord = CategoricalEncoder(strategy="ordinal", unknown_value=-1)
    res_ord = enc_ord.fit_transform(raw)
    assert res_ord.shape == (5, 1)
    assert res_ord.dtype == np.int64

    # Hash
    enc_hash = CategoricalEncoder(strategy="hash", hash_dim=8)
    res_hash = enc_hash.fit_transform(raw)
    assert res_hash.shape == (5, 8)


def test_tabular_encoder():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "salary": [50000.0, 60000.0, 75000.0, 90000.0],
        "city": ["NY", "SF", "NY", "LA"],
    })

    encoder = TabularEncoder()
    x = encoder.fit_transform(df)
    assert x.shape[0] == 4
    # age (1) + salary (1) + city (3 one-hot) = 5
    assert x.shape[1] == 5
    assert x.dtype == np.float32


# ==============================================================================
# 2. Graph Builders Unit Tests
# ==============================================================================

def test_knn_graph_builder():
    x = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0],
        [10.1, 10.1],
    ], dtype=np.float32)

    knn = KNNGraphBuilder(k=1, metric="euclidean", loop=False, bidirectional=True)
    edge_index, edge_attr = knn(x)

    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] > 0
    # Node 0 and Node 1 should be connected
    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert (0, 1) in edges and (1, 0) in edges
    # Node 2 and Node 3 should be connected
    assert (2, 3) in edges and (3, 2) in edges


def test_similarity_graph_builder():
    x = np.array([
        [1.0, 0.0],
        [0.99, 0.01],
        [0.0, 1.0],
    ], dtype=np.float32)

    sim = SimilarityGraphBuilder(threshold=0.9, metric="cosine", loop=False)
    edge_index, edge_attr = sim(x)

    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert (0, 1) in edges and (1, 0) in edges
    assert (0, 2) not in edges


def test_shared_entity_graph_builder():
    df = {
        "user_id": ["u1", "u2", "u3", "u4"],
        "device_id": ["d1", "d1", "d2", "d2"],
    }
    x = np.zeros((4, 2), dtype=np.float32)

    builder = SharedEntityGraphBuilder(entity_cols=["device_id"], loop=False)
    edge_index, edge_attr = builder(x, df_or_dict=df)

    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    # u1 (0) and u2 (1) share d1
    assert (0, 1) in edges and (1, 0) in edges
    # u3 (2) and u4 (3) share d2
    assert (2, 3) in edges and (3, 2) in edges
    # u1 (0) and u3 (2) do not share
    assert (0, 2) not in edges


def test_sequential_graph_builder():
    df = {
        "timestamp": [100, 102, 101, 200, 201],
        "session": ["s1", "s1", "s1", "s2", "s2"],
    }
    x = np.zeros((5, 2), dtype=np.float32)

    seq = SequentialGraphBuilder(order_col="timestamp", group_by_col="session", window_size=1)
    edge_index, edge_attr = seq(x, df_or_dict=df)

    edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    # s1 sequence ordered: 0 (ts=100) -> 2 (ts=101) -> 1 (ts=102)
    assert (0, 2) in edges
    assert (2, 1) in edges
    # s2 sequence: 3 (ts=200) -> 4 (ts=201)
    assert (3, 4) in edges
    # Cross session edges must not exist
    assert (1, 3) not in edges


# ==============================================================================
# 3. TableToGraph (Single-Table ETL) Tests
# ==============================================================================

def test_table_to_graph_dataframe():
    df = pd.DataFrame({
        "customer_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
        "age": [20, 25, 30, 45, 50, 55],
        "spend": [100.0, 150.0, 200.0, 500.0, 550.0, 600.0],
        "category": ["retail", "retail", "tech", "retail", "tech", "tech"],
        "churn": [0, 0, 0, 1, 1, 1],
    })

    etl = TableToGraph(
        target_col="churn",
        id_col="customer_id",
        edge_strategy="knn",
        edge_kwargs={"k": 2},
        train_ratio=0.5,
        test_ratio=0.5,
    )
    data = etl.fit_transform(df)

    assert isinstance(data, Data)
    assert data.x.shape[0] == 6
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.shape[1] > 0
    assert data.y.shape[0] == 6
    assert data.train_mask.shape[0] == 6
    assert data.test_mask.shape[0] == 6
    assert len(data.node_ids) == 6
    assert data.id_to_index["c1"] == 0


def test_table_to_graph_functional():
    data_dict = {
        "feat1": [1.0, 2.0, 3.0, 4.0],
        "feat2": [4.0, 3.0, 2.0, 1.0],
        "label": ["A", "B", "A", "B"],
    }
    data = table_to_graph(data_dict, target_col="label", edge_strategy="knn", edge_kwargs={"k": 1})
    assert isinstance(data, Data)
    assert data.x.shape == (4, 2)
    assert data.y.dtype == np.int64


def test_table_to_graph_from_csv():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "sample.csv")
        df = pd.DataFrame({
            "id": ["n1", "n2", "n3", "n4"],
            "v1": [10.0, 20.0, 30.0, 40.0],
            "v2": [1.0, 2.0, 3.0, 4.0],
            "y": [0, 1, 0, 1],
        })
        df.to_csv(csv_path, index=False)

        data = TableToGraph.from_csv(csv_path, target_col="y", id_col="id", edge_strategy="knn", edge_kwargs={"k": 1})
        assert isinstance(data, Data)
        assert data.x.shape[0] == 4
        assert data.node_ids == ["n1", "n2", "n3", "n4"]


# ==============================================================================
# 4. RelationalToGraph (Multi-Table ETL) Tests
# ==============================================================================

def test_relational_to_graph():
    users_df = pd.DataFrame({
        "user_id": ["u101", "u102", "u103"],
        "age": [22, 35, 48],
        "segment": ["bronze", "gold", "silver"],
    })
    products_df = pd.DataFrame({
        "prod_id": ["p1", "p2", "p3", "p4"],
        "price": [9.99, 49.99, 19.99, 99.99],
        "department": ["books", "elec", "books", "elec"],
    })
    purchases_df = pd.DataFrame({
        "u_id": ["u101", "u101", "u102", "u103", "unknown_user"],
        "p_id": ["p1", "p2", "p3", "p4", "p1"],
        "rating": [5.0, 4.0, 5.0, 3.0, 1.0],
    })

    etl = RelationalToGraph(
        id_cols={"user": "user_id", "product": "prod_id"},
        edge_cols={("user", "buys", "product"): ("u_id", "p_id")},
        edge_attr_cols={("user", "buys", "product"): ["rating"]},
    )

    hetero_data = etl.fit_transform(
        nodes={"user": users_df, "product": products_df},
        edges={("user", "buys", "product"): purchases_df},
    )

    assert isinstance(hetero_data, HeteroData)
    # Check node tables
    assert hetero_data["user"].x.shape[0] == 3
    assert hetero_data["product"].x.shape[0] == 4

    # Check edges (unknown_user should be filtered out)
    edge_index = hetero_data["user", "buys", "product"].edge_index
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 4  # 4 valid purchases

    # Check edge attributes
    edge_attr = hetero_data["user", "buys", "product"].edge_attr
    assert edge_attr.shape == (4, 1)

    # Check ID mapping consistency
    assert hetero_data.id_maps["user"]["u101"] == 0
    assert hetero_data.inverse_id_maps["user"][0] == "u101"


# ==============================================================================
# 5. End-to-End ETL to GNN Training Integration
# ==============================================================================

def test_etl_to_node_classifier_integration():
    """Verify that a DataFrame converted via TableToGraph can directly train a NodeClassifier."""
    df = pd.DataFrame({
        "feat_a": np.random.randn(30).astype("float32"),
        "feat_b": np.random.randn(30).astype("float32"),
        "category": (["cat1", "cat2", "cat3"] * 10),
        "target": (np.random.randint(0, 2, size=30)),
    })

    # 1. ETL: Tabular -> Graph Data
    data = table_to_graph(
        df,
        target_col="target",
        edge_strategy="knn",
        edge_kwargs={"k": 3},
        train_ratio=0.7,
        test_ratio=0.3,
    )

    # 2. Train NodeClassifier in 3 lines of code!
    clf = NodeClassifier(backbone="gcn", hidden_channels=16, num_layers=2)
    clf.fit(data, epochs=2, verbose=0)

    # 3. Predict & evaluate
    metrics = clf.evaluate(data, mask="test_mask")
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
