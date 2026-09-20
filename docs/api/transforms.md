# Graph Transforms

`k3_node.transforms` provides 62+ deterministic and stochastic transformations for graphs, 3D point clouds, meshes, and node features.

Transforms can be chained using `Compose` and passed to datasets as `transform` (applied dynamically on each get) or `pre_transform` (applied once during offline processing).

```python
from k3_node.transforms import Compose, ToUndirected, AddSelfLoops, NormalizeFeatures

transform = Compose([
    ToUndirected(),
    AddSelfLoops(),
    NormalizeFeatures(),
])

data = transform(data)
```

---

## General Transforms

Transforms operating on node/edge feature matrices, masks, and splits:

| Transform | Description |
|---|---|
| `Compose` | Sequentially chains multiple transforms together. |
| `ComposeFilters` | Filters graphs based on Boolean predicate functions. |
| `NormalizeFeatures` | Row-normalizes node features ($L_1$ or $L_2$ norm). |
| `Constant` | Adds a constant scalar or vector value to node features. |
| `RandomNodeSplit` | Generates train/validation/test node masks randomly or with fixed counts. |
| `RandomLinkSplit` | Splits edge indices into train/val/test edges with negative sampling. |
| `NodePropertySplit` | Partitions nodes based on continuous or categorical properties. |
| `IndexToMask` | Converts an array of indices into a Boolean mask. |
| `MaskToIndex` | Converts a Boolean mask into an array of active indices. |
| `SVDFeatureReduction` | Dimensionality reduction on node feature matrices via SVD. |
| `RemoveTrainingClasses` | Removes specific classes from the training set for open-world settings. |
| `Pad` | Pads node feature matrices and adjacency representations to a fixed budget. |
| `ToSparseTensor` | Converts `edge_index` to sparse adjacency matrix format. |
| `ToDevice` | Transfers tensor attributes to a specific compute device. |

::: k3_node.transforms.Compose

::: k3_node.transforms.NormalizeFeatures

---

## Graph Structural & Spectral Transforms

Transforms modifying graph topology, adding self-loops, computing positional encodings, and rewiring edges:

| Transform | Description |
|---|---|
| `ToUndirected` | Converts a directed graph into an undirected graph by adding reciprocal edges. |
| `AddSelfLoops` | Adds self-loops $(i, i)$ to all nodes. |
| `AddRemainingSelfLoops` | Adds self-loops only to nodes that do not already have one. |
| `RemoveSelfLoops` | Removes all self-loops from the graph. |
| `RemoveIsolatedNodes` | Prunes isolated nodes without any incident edges. |
| `RemoveDuplicatedEdges` | Deduplicates multi-edges in the graph. |
| `TwoHop` | Adds edges connecting nodes that are 2 hops apart. |
| `LineGraph` | Constructs the line graph where original edges become nodes. |
| `LargestConnectedComponents` | Restricts the graph to its largest connected component(s). |
| `VirtualNode` | Adds an auxiliary virtual super-node connected to all nodes. |
| `GCNNorm` | Computes standard GCN symmetric normalized adjacency matrix $\mathbf{\tilde{D}}^{-1/2}\mathbf{\tilde{A}}\mathbf{\tilde{D}}^{-1/2}$. |
| `GDC` | Generalized Graph Diffusion (PageRank or Heat kernel diffusion) for graph denoising and rewiring. |
| `SIGN` | Precomputes multi-hop diffused feature aggregations for Scalable Inception Graph Neural Networks. |
| `AddLaplacianEigenvectorPE` | Computes Laplacian eigenvector positional encodings ($k$ smallest non-trivial eigenvectors). |
| `AddRandomWalkPE` | Computes Random Walk structural positional encodings (landing probabilities $RW_{i,i}^k$). |
| `AddGPSE` | Computes 20-dimensional Graph Positional and Structural Encodings. |
| `FeaturePropagation` | Reconstructs missing node features via Dirichlet energy diffusion. |
| `HalfHop` | Slow-fast graph rewiring for over-smoothing and over-squashing mitigation. |
| `OneHotDegree` | Computes one-hot encodings of node degrees. |
| `TargetIndegree` | Appends target node in-degrees as edge features. |
| `LocalDegreeProfile` | Extracts statistical degree distribution summaries for local neighborhoods. |
| `KNNGraph` | Constructs a $k$-nearest-neighbor graph from spatial coordinates. |
| `RadiusGraph` | Constructs an $\epsilon$-ball radius graph from spatial coordinates. |
| `ToDense` | Converts sparse graph representation into dense adjacency and feature matrices. |

::: k3_node.transforms.ToUndirected

::: k3_node.transforms.AddSelfLoops

::: k3_node.transforms.AddLaplacianEigenvectorPE

::: k3_node.transforms.AddRandomWalkPE

---

## Spatial & Geometric Transforms

Transforms for 3D point clouds, molecular coordinates, and triangle meshes:

| Transform | Description |
|---|---|
| `Cartesian` | Saves relative Cartesian coordinates $\mathbf{p}_j - \mathbf{p}_i$ as edge attributes. |
| `LocalCartesian` | Relative Cartesian coordinates projected onto local coordinate frames. |
| `Distance` | Computes Euclidean distances $\|\mathbf{p}_j - \mathbf{p}_i\|_2$ as edge attributes. |
| `Polar` | Converts 2D/3D relative offsets to polar/cylindrical coordinates. |
| `Spherical` | Converts 3D relative offsets to spherical coordinates $(r, \theta, \phi)$. |
| `PointPairFeatures` | Computes Point Pair Features (PPF) using surface normal vectors. |
| `Center` | Centers coordinates around origin $(\sum \mathbf{p} = \mathbf{0})$. |
| `NormalizeScale` | Rescales coordinates into the unit sphere $[-1, 1]$. |
| `NormalizeRotation` | Aligns principal axes of point clouds via eigenvectors of inertia tensor. |
| `RandomRotate` | Randomly rotates 3D point cloud / mesh coordinates. |
| `RandomScale` | Randomly scales 3D coordinates by a factor sampled uniformly from $[\min, \max]$. |
| `RandomTranslate` | Randomly offsets coordinates with bounded translation noise. |
| `RandomJitter` | Adds Gaussian or uniform jitter noise to coordinates. |
| `RandomFlip` | Randomly mirrors coordinates along chosen axes. |
| `SamplePoints` | Uniformly samples a fixed number of points from triangle meshes. |
| `FixedPoints` | Subsamples or resamples point clouds to a fixed number of points. |
| `GenerateMeshNormals` | Computes per-face and per-vertex surface normal vectors from triangle faces. |
| `FaceToEdge` | Converts mesh triangular faces `face` into undirected edge indices `edge_index`. |
| `Delaunay` | Computes Delaunay triangulation for 2D/3D point clouds. |
| `GridSampling` | Voxel grid spatial downsampling of point clouds. |
| `ToSLIC` | Generates superpixels from images using Simple Linear Iterative Clustering. |

