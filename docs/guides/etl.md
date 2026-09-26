# Tabular-to-Graph ETL Guide

Real-world datasets often originate as tabular files (CSVs, Parquet, pandas/polars DataFrames, SQL databases) rather than pre-constructed graph topologies.

K3-Node provides an end-to-end **Tabular-to-Graph ETL** API under `k3_node.etl` to transform single tables or multi-table relational databases into graph `Data` or `HeteroData` objects ready for GNN training.

---

## 1. Single Table to Graph (`TableToGraph`)

`TableToGraph` (alias `TabularToGraph`) converts a DataFrame or CSV file into a `k3_node.data.Data` object. It automatically encodes numerical and categorical features, constructs graph edges according to your chosen strategy, maps IDs, and generates train/val/test splits.

### Example: Customer Churn with k-NN Graph

```python
import pandas as pd
from k3_node.etl import TableToGraph
from k3_node.tasks import NodeClassifier

# 1. Load raw tabular DataFrame
df = pd.DataFrame({
    "customer_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
    "age": [25, 30, 45, 50, 22, 60],
    "monthly_spend": [120.0, 150.0, 300.0, 320.0, 110.0, 400.0],
    "plan_type": ["basic", "pro", "enterprise", "enterprise", "basic", "enterprise"],
    "churn": [0, 0, 1, 1, 0, 1],
})

# 2. Convert table to graph in one line
etl = TableToGraph(
    target_col="churn",
    id_col="customer_id",
    edge_strategy="knn",
    edge_kwargs={"k": 2, "metric": "cosine"},
    train_ratio=0.7,
    test_ratio=0.3,
)
data = etl.fit_transform(df)

# 3. Directly train a NodeClassifier!
clf = NodeClassifier(backbone="gcn", hidden_channels=32, num_layers=2)
clf.fit(data, epochs=20, lr=0.01)

metrics = clf.evaluate(data, mask="test_mask")
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
```

### Loading Directly from CSV
```python
from k3_node.etl import TableToGraph

data = TableToGraph.from_csv(
    "data/transactions.csv",
    target_col="is_fraud",
    id_col="tx_id",
    edge_strategy="knn",
    edge_kwargs={"k": 5},
)
```

---

## 2. Graph Topology Strategies

K3-Node provides four built-in topology generation builders:

### 1. k-Nearest Neighbors (`"knn"`)
Connects each row to its $k$ closest rows based on feature space distance.
```python
etl = TableToGraph(
    edge_strategy="knn",
    edge_kwargs={"k": 5, "metric": "cosine", "bidirectional": True},
)
```
- Metrics supported: `"cosine"`, `"euclidean"`, `"manhattan"`.

### 2. Pairwise Similarity Threshold (`"similarity"`)
Connects rows whose pairwise similarity exceeds a threshold $\tau$.
```python
etl = TableToGraph(
    edge_strategy="similarity",
    edge_kwargs={"threshold": 0.8, "metric": "cosine"},
)
```
- Metrics supported: `"cosine"`, `"rbf"`.

### 3. Shared Entity / Bipartite Projection (`"shared_entity"`)
Connects rows that share identical values in one or more categorical identifier columns (e.g., users sharing the same IP address, device ID, or cluster).
```python
etl = TableToGraph(
    edge_strategy="shared_entity",
    edge_kwargs={"entity_cols": ["ip_address", "device_fingerprint"], "max_degree": 50},
)
```

### 4. Sequential / Chronological (`"sequential"`)
Connects rows sequentially in order of a timestamp or sequence index, optionally grouped by an entity (e.g. clickstreams per user session).
```python
etl = TableToGraph(
    edge_strategy="sequential",
    edge_kwargs={"order_col": "timestamp", "group_by_col": "session_id", "window_size": 2},
)
```

---

## 3. Multi-Table Relational ETL (`RelationalToGraph`)

For relational databases with foreign keys, `RelationalToGraph` converts multiple tables into a `k3_node.data.HeteroData` graph. It handles entity ID mapping (e.g. string UUIDs $\rightarrow$ contiguous integer indices), node feature encoding, edge attributes, and filters orphaned edges.

```python
import pandas as pd
from k3_node.etl import RelationalToGraph

# 1. Node tables
users_df = pd.DataFrame({
    "user_id": ["u1", "u2", "u3"],
    "age": [25, 40, 32],
    "segment": ["consumer", "enterprise", "consumer"],
})
items_df = pd.DataFrame({
    "item_id": ["i1", "i2", "i3", "i4"],
    "price": [15.0, 99.0, 45.0, 120.0],
    "category": ["apparel", "electronics", "apparel", "electronics"],
})

# 2. Edge / Interaction table
ratings_df = pd.DataFrame({
    "user_id": ["u1", "u1", "u2", "u3"],
    "item_id": ["i1", "i2", "i3", "i4"],
    "rating": [5.0, 4.0, 3.0, 5.0],
})

# 3. ETL conversion
etl = RelationalToGraph(
    id_cols={"user": "user_id", "item": "item_id"},
    edge_cols={("user", "rates", "item"): ("user_id", "item_id")},
    edge_attr_cols={("user", "rates", "item"): ["rating"]},
)

hetero_data = etl.fit_transform(
    nodes={"user": users_df, "item": items_df},
    edges={("user", "rates", "item"): ratings_df},
)

print(hetero_data["user"].x.shape)                     # (3, 3)
print(hetero_data["item"].x.shape)                     # (4, 3)
print(hetero_data["user", "rates", "item"].edge_index) # (2, 4)
print(hetero_data["user", "rates", "item"].edge_attr)  # (4, 1)
```

### Accessing Original Entity IDs
The mapping from raw business identifiers to node indices is stored directly on the resulting `HeteroData` object:
```python
# Convert raw user ID "u1" to graph node index
user_node_idx = hetero_data.id_maps["user"]["u1"]

# Convert graph node index 0 back to raw user ID
raw_user_id = hetero_data.inverse_id_maps["user"][0]
```

---

## 4. Standalone Encoders

If you need fine-grained control over column encoding:
- `NumericalEncoder(strategy="standard"|"minmax"|"log1p", impute_strategy="mean"|"median"|"zero")`
- `CategoricalEncoder(strategy="onehot"|"ordinal"|"hash", handle_unknown="ignore")`
- `TabularEncoder(column_encoders={...})`
