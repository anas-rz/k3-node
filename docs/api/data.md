# Graph Data Structures

`k3_node.data` provides graph data structures, datasets, and storage abstractions compatible with NumPy, PyTorch, TensorFlow, and JAX tensors.

---

## Graph Containers

### Data
`Data` represents a single homogeneous graph, holding node attributes, edge indices, edge attributes, and graph-level properties.

```python
import numpy as np
from k3_node.data import Data

# Graph with 3 nodes and 4 directed edges
x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
edge_index = np.array([[0, 1, 1, 2],
                       [1, 0, 2, 1]], dtype=np.int64)

data = Data(x=x, edge_index=edge_index)
print(data.num_nodes)  # 3
print(data.num_edges)  # 4
```

::: k3_node.data.Data

---

### HeteroData
`HeteroData` represents heterogeneous graphs with multiple node types and edge types.

```python
from k3_node.data import HeteroData

data = HeteroData()
data['paper'].x = paper_features
data['author'].x = author_features
data['author', 'writes', 'paper'].edge_index = author_writes_paper_edges

print(data.node_types)  # ['paper', 'author']
print(data.edge_types)  # [('author', 'writes', 'paper')]
```

::: k3_node.data.HeteroData

---

### Batch & HeteroBatch
`Batch` combines multiple `Data` objects into a single giant disjoint graph with diagonal adjacency block structure and a `batch` vector tracking graph membership.

```python
from k3_node.data import Batch, Data

graph1 = Data(x=x1, edge_index=edge_index1)
graph2 = Data(x=x2, edge_index=edge_index2)

batch = Batch.from_data_list([graph1, graph2])
print(batch.batch)       # [0, 0, 0, ..., 1, 1, 1]
print(batch.num_graphs)  # 2

# Deconstruct back into individual graphs
data_list = batch.to_data_list()
```

::: k3_node.data.Batch

---

### TemporalData & HypergraphData

- `TemporalData`: Encapsulates continuous-time dynamic interaction streams with timestamps (`t`), source nodes (`src`), destination nodes (`dst`), and edge attributes (`msg`).
- `HypergraphData`: Encapsulates hypergraphs where hyperedges can connect an arbitrary number of nodes.

::: k3_node.data.TemporalData

::: k3_node.data.HypergraphData

---

## Datasets

### Dataset & InMemoryDataset
- `Dataset`: Base class for graph datasets, supporting raw file downloading, processing, and indexing.
- `InMemoryDataset`: Dataset that fits completely into memory, with fast serialization/deserialization.
- `OnDiskDataset`: Out-of-core dataset backed by key-value stores (`Database`, `SQLiteDatabase`, `RocksDatabase`).

```python
from k3_node.data import InMemoryDataset

class MyGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Process raw files and save graphs
        ...
```

::: k3_node.data.InMemoryDataset

---

## Remote & Large-Scale Stores

- `FeatureStore`: Standard interface for external or out-of-memory node/edge feature storage.
- `GraphStore`: Standard interface for graph topology and edge indexing.
- `SQLiteDatabase`: SQLite3-backed persistent graph and attribute storage.
- `RocksDatabase`: RocksDB-backed key-value storage.

---

## Download & Extraction Utilities

- `download_url(url, folder, filename=None)`: Downloads a file over HTTP/HTTPS with progress bar.
- `download_google_url(id, folder, filename)`: Downloads large pre-trained checkpoints or datasets directly from Google Drive.
- `extract_zip(path, folder)` / `extract_tar(path, folder)` / `extract_gz(path, folder)` / `extract_bz2(path, folder)`: Archive extraction helpers.
- `makedirs(path)`: Recursive directory creation helper.

