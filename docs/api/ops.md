# Graph Utilities & Ops

The `k3_node.ops` module collects the low-level graph-algebra, sparse-matrix, structure-checking, synthetic-graph-generation, and cheminformatics helper functions used throughout K3-Node's layers, models, and datasets. It's a superset of the legacy `k3_node.utils` namespace (kept for backward compatibility) — new code should import from `k3_node.ops`.

```python
from k3_node import ops as k3_ops
```

---

## Graph Algebra

Dense/sparse adjacency-matrix normalization utilities, mirroring `torch_geometric`/Spektral-style graph filters — mainly used internally by spectral layers (`ChebConv`, `GCN2Conv`, ...).

### degree_matrix
::: k3_node.ops.conv.degree_matrix

### degree_power
::: k3_node.ops.conv.degree_power

### normalized_adjacency
::: k3_node.ops.conv.normalized_adjacency

### normalized_laplacian
::: k3_node.ops.conv.normalized_laplacian

### laplacian
::: k3_node.ops.conv.laplacian

### gcn_filter
::: k3_node.ops.conv.gcn_filter

### normalize_A
::: k3_node.ops.graph.normalize_A

### degrees
::: k3_node.ops.graph.degrees

### get_source_target
::: k3_node.ops.graph.get_source_target

---

## Sparse / Dense Matrix Operations

Backend-agnostic matrix-multiply helpers that transparently handle mixed sparse/dense and batched ("modal") tensors.

### dot
::: k3_node.ops.matmul.dot

### mixed_mode_dot
::: k3_node.ops.matmul.mixed_mode_dot

### modal_dot
::: k3_node.ops.matmul.modal_dot

### polyval
::: k3_node.ops.numpy.polyval

### get_unique
::: k3_node.ops.numpy.get_unique

---

## Graph Structure Utilities

Edge-index bookkeeping helpers, mirroring `torch_geometric.utils`.

### coalesce
::: k3_node.utils.graph.coalesce

### subgraph
::: k3_node.utils.graph.subgraph

### edge_index_to_adjacency_matrix
::: k3_node.utils.graph.edge_index_to_adjacency_matrix

### contains_isolated_nodes
::: k3_node.utils.graph.contains_isolated_nodes

### has_self_loops
::: k3_node.utils.graph.has_self_loops

### is_undirected
::: k3_node.utils.graph.is_undirected

---

## Synthetic Graph Generators

Reference-graph generators used in tests and tutorials.

### erdos_renyi_graph
::: k3_node.utils.random.erdos_renyi_graph

### barabasi_albert_graph
::: k3_node.utils.random.barabasi_albert_graph

### stochastic_blockmodel_graph
::: k3_node.utils.random.stochastic_blockmodel_graph

---

## Cheminformatics (SMILES / RDKit)

Conversions between SMILES strings, RDKit `Mol` objects, and K3-Node `Data` graphs — used by the molecular datasets (`MoleculeNet`, `QM9`, `QM7`) and chemistry models.

### from_smiles
::: k3_node.utils.smiles.from_smiles

### to_smiles
::: k3_node.utils.smiles.to_smiles

### from_rdmol
::: k3_node.utils.smiles.from_rdmol

### to_rdmol
::: k3_node.utils.smiles.to_rdmol

---

## Neural Network Ops

### segment_softmax
Numerically-stable, segment-wise (per-graph or per-node-neighborhood) softmax — the core primitive behind every attention-based conv layer (`GATConv`, `TransformerConv`, ...).

::: k3_node.utils.keras.segment_softmax
