# Graph Data Loaders

`k3_node.loader` provides multi-backend graph data loaders and sampling utilities for large-scale graph neural networks.

---

## Mini-Batch Graph Loaders

| Loader | Description |
|---|---|
| `DataLoader` | Batches multiple homogeneous / heterogeneous `Data` objects into a single disjoint batch (`Batch`). |
| `DenseDataLoader` | Stacks multiple graph adjacency matrices into dense tensors of shape `(batch_size, num_nodes, num_nodes)`. |
| `DataListLoader` | Yields lists of `Data` objects without merging into a single disjoint batch (useful for multi-GPU training). |
| `TemporalDataLoader` | Successive temporal event window mini-batch loader for continuous-time dynamic graphs. |
| `ZipLoader` | Combines multiple data loaders into synchronized tuples. |

```python
from k3_node.loader import DataLoader
from k3_node.data import Data

dataset = [Data(x=..., edge_index=...) for _ in range(100)]
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    out = model(batch.x, batch.edge_index, batch=batch.batch)
```

::: k3_node.loader.DataLoader

---

## Neighbor & Subgraph Samplers

Scalable graph neural network training on large graphs (millions of nodes):

| Sampler | Description |
|---|---|
| `NeighborLoader` | Multi-hop neighbor sampling for mini-batch training without C++ dependencies. |
| `LinkNeighborLoader` | Link-centric neighbor sampling with positive and negative edge supervision. |
| `NodeLoader` | Mini-batch sampling from specified node indices. |
| `LinkLoader` | Mini-batch sampling from link/edge information. |
| `HGTLoader` | Heterogeneous Graph Transformer balanced neighbor sampling across types. |
| `ShaDowKHopSampler` | Decoupled shallow ego-network subgraph extractor. |
| `NeighborSampler` | Classical layer-by-layer bipartite neighbor sampler. |

```python
from k3_node.loader import NeighborLoader

loader = NeighborLoader(
    data,
    num_neighbors=[15, 10],  # 15 neighbors at 1st hop, 10 at 2nd hop
    batch_size=128,
    input_nodes=data.train_mask,
)

for batch in loader:
    pred = model(batch.x, batch.edge_index)[:batch.batch_size]
```

::: k3_node.loader.NeighborLoader

---

## Graph Partitioning & SAINT Samplers

| Sampler | Description |
|---|---|
| `ClusterData` | Graph partitioner using METIS or pure Python/BFS fallback. |
| `ClusterLoader` | Merges partitioned subgraphs into mini-batches. |
| `GraphSAINTNodeSampler` | Node-budget random subgraph sampler. |
| `GraphSAINTEdgeSampler` | Edge-probability random subgraph sampler. |
| `GraphSAINTRandomWalkSampler` | Random-walk-based subgraph sampler. |
| `RandomNodeLoader` | Random node partition loader for large graphs. |

```python
from k3_node.loader import ClusterData, ClusterLoader

cluster_data = ClusterData(data, num_parts=128)
loader = ClusterLoader(cluster_data, batch_size=32, shuffle=True)
```

::: k3_node.loader.ClusterLoader

---

## Performance & Utility Mixins

| Component | Description |
|---|---|
| `PrefetchLoader` | Asynchronous host-to-device memory prefetcher. |
| `CachedLoader` | In-memory mini-batch cache across training epochs. |
| `DynamicBatchSampler` | Dynamic node/edge budget mini-batch sampler. |
| `ImbalancedSampler` | Class-frequency weighted random sampler for class imbalance. |
| `AffinityMixin` | CPU worker core affinitization context manager. |
| `MultithreadingMixin` | Worker subprocess thread count configuration. |
| `LogMemoryMixin` | Worker RSS memory consumption logger. |

