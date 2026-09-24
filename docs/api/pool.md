# Pooling Layers

The `k3_node.layers.pool` module provides global readouts, hierarchical graph coarsening, cluster-based pooling, and spatial graph construction operators.

---

## Global Graph Readout

### global_add_pool
::: k3_node.layers.pool.global_add_pool

### global_mean_pool
::: k3_node.layers.pool.global_mean_pool

### global_max_pool
::: k3_node.layers.pool.global_max_pool

---

## Hierarchical Node Pooling

### TopKPooling
::: k3_node.layers.pool.TopKPooling

### SAGPooling
::: k3_node.layers.pool.SAGPooling

### EdgePooling
::: k3_node.layers.pool.EdgePooling

### ASAPooling
::: k3_node.layers.pool.ASAPooling

### PANPooling
::: k3_node.layers.pool.PANPooling

### MemPooling
::: k3_node.layers.pool.MemPooling

### ClusterPooling
::: k3_node.layers.pool.ClusterPooling

---

## Spatial & Neighborhood Pooling

### avg_pool_neighbor_x
::: k3_node.layers.pool.avg_pool_neighbor_x

### max_pool_neighbor_x
::: k3_node.layers.pool.max_pool_neighbor_x

### voxel_grid
::: k3_node.layers.pool.voxel_grid

### fps
::: k3_node.layers.pool.fps

### graclus
::: k3_node.layers.pool.graclus

---

## Graph Construction

### radius_graph
::: k3_node.layers.pool.radius_graph

### knn_graph
::: k3_node.layers.pool.knn_graph

---

## Unpooling

Interpolates features back from a coarsened point set/graph onto a denser one, mirroring `torch_geometric.nn.unpool`.

### knn_interpolate
::: k3_node.layers.unpool.knn_interpolate

---

## Regularization Functionals

Differentiable graph-pooling regularizers from `k3_node.layers.functional`, typically added as auxiliary loss terms alongside a pooling layer (e.g. `DMoNPooling`, `MinCutPooling`) to encourage balanced, well-separated cluster assignments.

### bro
::: k3_node.layers.functional.bro

### gini
::: k3_node.layers.functional.gini

