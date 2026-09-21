# K3-Node Layers Multi-Backend Verification Checklist

> Tracking layer-by-layer learning and training verification across PyTorch, TensorFlow, and JAX backends.

## Overall Progress Summary

| Category | Total | Tested | Passed | Failed | Base/Abstract/Vendor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Convolution Layers (`k3_node.layers.conv`) | 75 | 66 | 66 | 0 | 9 |
| Normalization Layers (`k3_node.layers.norm`) | 11 | 9 | 9 | 0 | 2 |
| Pooling Layers (`k3_node.layers.pool`) | 11 | 9 | 9 | 0 | 2 |
| Aggregation Layers (`k3_node.layers.aggr`) | 28 | 27 | 27 | 0 | 1 |
| Dense / Linear Layers (`k3_node.layers.dense`) | 9 | 7 | 7 | 0 | 2 |
| Attention Layers (`k3_node.layers.attention`) | 6 | 6 | 6 | 0 | 0 |
| Knowledge Graph Embedding (`k3_node.layers.kge`) | 5 | 4 | 4 | 0 | 1 |
---

## Convolution Layers (`k3_node.layers.conv`)

| Layer Name | Module | PyTorch | TensorFlow | JAX | Learning Verified | Notes / Fixes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `AGNNConv` | `k3_node.layers.conv.agnn_conv` | [x] PASS (1.48->0.60) | [x] PASS (1.87->0.63) | [x] PASS (1.75->0.58) | [x] |  |
| `AntiSymmetricConv` | `k3_node.layers.conv.antisymmetric_conv` | [x] PASS (2.31->0.59) | [x] PASS (1.60->0.30) | [x] PASS (2.08->0.59) | [x] |  |
| `APPNP` | `k3_node.layers.conv.appnp` | [x] PASS (0.75->0.36) | [x] PASS (1.20->0.59) | [x] PASS (1.52->0.75) | [x] |  |
| `APPNPConv` | `k3_node.layers.conv.appnp_conv` | [x] PASS (4.89->1.04) | [x] PASS (2.99->0.98) | [x] PASS (1.98->0.62) | [x] |  |
| `ARMAConv` | `k3_node.layers.conv.arma_conv` | [x] PASS (1.17->0.66) | [x] PASS (1.94->0.79) | [x] PASS (0.82->0.41) | [x] |  |
| `CGConv` | `k3_node.layers.conv.cg_conv` | [x] PASS (3.51->0.65) | [x] PASS (2.94->0.69) | [x] PASS (3.17->1.05) | [x] |  |
| `ChebConv` | `k3_node.layers.conv.cheb_conv` | [x] PASS (3.93->0.65) | [x] PASS (4.57->0.61) | [x] PASS (3.53->0.78) | [x] |  |
| `ClusterGCNConv` | `k3_node.layers.conv.cluster_gcn_conv` | [x] PASS (1.87->0.64) | [x] PASS (5.35->0.63) | [x] PASS (4.19->0.47) | [x] |  |
| `CrystalConv` | `k3_node.layers.conv.crystal_conv` | [x] PASS (6.19->1.13) | [x] PASS (4.10->0.40) | [x] PASS (3.58->0.25) | [x] |  |
| `CuGraphGATConv` | `k3_node.layers.conv.cugraph` | [-] SKIP | [-] SKIP | [-] SKIP | [-] | cuGraph vendor layer (requires GPU cuGraph bindings) |
| `CuGraphSAGEConv` | `k3_node.layers.conv.cugraph` | [-] SKIP | [-] SKIP | [-] SKIP | [-] | cuGraph vendor layer (requires GPU cuGraph bindings) |
| `DiffusionConv` | `k3_node.layers.conv.diffusion_conv` | [x] PASS (1.19->0.72) | [x] PASS (1.49->1.23) | [x] PASS (1.53->0.66) | [x] |  |
| `DirGNNConv` | `k3_node.layers.conv.dir_gnn_conv` | [x] PASS (2.18->0.81) | [x] PASS (2.87->1.04) | [x] PASS (2.79->1.04) | [x] |  |
| `DNAConv` | `k3_node.layers.conv.dna_conv` | [x] PASS (1.17->0.57) | [x] PASS (1.69->0.77) | [x] PASS (0.82->0.39) | [x] |  |
| `DynamicEdgeConv` | `k3_node.layers.conv.edge_conv` | [x] PASS (1.82->0.60) | [x] PASS (1.32->0.68) | [x] PASS (1.88->0.86) | [x] |  |
| `EdgeConv` | `k3_node.layers.conv.edge_conv` | [x] PASS (1.18->0.65) | [x] PASS (1.96->0.72) | [x] PASS (1.81->0.93) | [x] |  |
| `EGConv` | `k3_node.layers.conv.eg_conv` | [x] PASS (10.46->0.75) | [x] PASS (3.41->0.73) | [x] PASS (2.35->0.38) | [x] |  |
| `FAConv` | `k3_node.layers.conv.fa_conv` | [x] PASS (1.43->0.77) | [x] PASS (1.45->0.84) | [x] PASS (1.16->0.61) | [x] |  |
| `FeaStConv` | `k3_node.layers.conv.feast_conv` | [x] PASS (1.59->0.70) | [x] PASS (1.45->0.60) | [x] PASS (1.44->0.91) | [x] |  |
| `FiLMConv` | `k3_node.layers.conv.film_conv` | [x] PASS (3.07->1.22) | [x] PASS (3.50->1.00) | [x] PASS (4.58->0.97) | [x] |  |
| `FusedGATConv` | `k3_node.layers.conv.gat_conv` | [x] PASS (1.56->0.77) | [x] PASS (1.87->1.04) | [x] PASS (2.13->1.05) | [x] |  |
| `GATConv` | `k3_node.layers.conv.gat_conv` | [x] PASS (1.65->0.67) | [x] PASS (1.47->0.71) | [x] PASS (1.38->0.68) | [x] |  |
| `GatedGraphConv` | `k3_node.layers.conv.gated_graph_conv` | [x] PASS (1.08->0.64) | [x] PASS (1.00->0.53) | [x] PASS (1.70->0.86) | [x] |  |
| `GATv2Conv` | `k3_node.layers.conv.gatv2_conv` | [x] PASS (1.58->0.91) | [x] PASS (2.36->0.70) | [x] PASS (1.95->0.92) | [x] |  |
| `GraphConvolution` | `k3_node.layers.conv.gcn` | [x] PASS (4.26->0.71) | [x] PASS (3.64->0.88) | [x] PASS (2.71->0.91) | [x] |  |
| `GCN2Conv` | `k3_node.layers.conv.gcn2_conv` | [x] PASS (2.94->0.82) | [x] PASS (2.28->0.93) | [x] PASS (2.01->0.71) | [x] |  |
| `GCNConv` | `k3_node.layers.conv.gcn_conv` | [x] PASS (1.36->0.37) | [x] PASS (1.60->0.78) | [x] PASS (2.81->0.85) | [x] |  |
| `GENConv` | `k3_node.layers.conv.gen_conv` | [x] PASS (1.41->0.55) | [x] PASS (0.89->0.28) | [x] PASS (2.49->0.48) | [x] |  |
| `GeneralConv` | `k3_node.layers.conv.general_conv` | [x] PASS (11.11->2.51) | [x] PASS (2.76->0.82) | [x] PASS (17.71->2.07) | [x] |  |
| `GINConv` | `k3_node.layers.conv.gin_conv` | [x] PASS (3.36->0.75) | [x] PASS (2.73->1.14) | [x] PASS (5.82->1.64) | [x] |  |
| `GINEConv` | `k3_node.layers.conv.gin_conv` | [x] PASS (2.45->0.67) | [x] PASS (7.84->1.49) | [x] PASS (1.92->0.61) | [x] |  |
| `GMMConv` | `k3_node.layers.conv.gmm_conv` | [x] PASS (2.10->0.61) | [x] PASS (2.68->0.87) | [x] PASS (2.04->0.51) | [x] |  |
| `GPSConv` | `k3_node.layers.conv.gps_conv` | [x] PASS (8.23->0.46) | [x] PASS (6.60->0.40) | [x] PASS (10.43->1.16) | [x] |  |
| `GraphAttention` | `k3_node.layers.conv.graph_attention` | [x] PASS (2.38->1.49) | [x] PASS (1.80->0.97) | [x] PASS (1.20->0.62) | [x] |  |

...; jax error: Unable to automatically build the model. Please bu... |
| `GraphConv` | `k3_node.layers.conv.graph_conv` | [x] PASS (2.51->0.76) | [x] PASS (5.42->0.75) | [x] PASS (2.40->0.43) | [x] |  |
| `GravNetConv` | `k3_node.layers.conv.gravnet_conv` | [x] PASS (3.22->1.19) | [x] PASS (2.46->1.22) | [x] PASS (2.74->1.23) | [x] |  |
| `HANConv` | `k3_node.layers.conv.han_conv` | [-] SKIP | [-] SKIP | [-] SKIP | [-] | Heterogeneous GNN layer (tested in hetero suite) |
| `HEATConv` | `k3_node.layers.conv.heat_conv` | [-] SKIP | [-] SKIP | [-] SKIP | [-] | Heterogeneous GNN layer (tested in hetero suite) |
| `HeteroConv` | `k3_node.layers.conv.hetero_conv` | [-] SKIP | [-] SKIP | [-] SKIP | [-] | Heterogeneous GNN layer (tested in hetero suite) |
| `HGTConv` | `k3_node.layers.conv.hgt_conv` | [-] SKIP | [-] SKIP | [-] SKIP | [-] | Heterogeneous GNN layer (tested in hetero suite) |
| `HypergraphConv` | `k3_node.layers.conv.hypergraph_conv` | [x] PASS (0.93->0.73) | [x] PASS (0.96->0.67) | [x] PASS (1.00->0.81) | [x] | |
| `LEConv` | `k3_node.layers.conv.le_conv` | [x] PASS (19.28->1.42) | [x] PASS (10.81->0.99) | [x] PASS (11.71->1.12) | [x] | |
| `LGConv` | `k3_node.layers.conv.lg_conv` | [x] PASS (2.25->0.95) | [x] PASS (2.89->1.26) | [x] PASS (2.87->0.96) | [x] | |
| `MeshCNNConv` | `k3_node.layers.conv.meshcnn_conv` | [x] PASS (9.12->1.65) | [x] PASS (5.69->0.94) | [x] PASS (10.16->2.01) | [x] | |
| `MessagePassing` | `k3_node.layers.conv.message_passing` | [-] SKIP | [-] SKIP | [-] SKIP | [-] | Base/Abstract class |
  File "/home/a...; tensorflow error: WARNING: All log messages before absl::InitializeL...; jax error: Traceback (most recent call last):
  File "/home/a... |
| `MFConv` | `k3_node.layers.conv.mf_conv` | [x] PASS (6.97->0.70) | [x] PASS (4.03->0.42) | [x] PASS (4.04->0.42) | [x] |  |
| `MixHopConv` | `k3_node.layers.conv.mixhop_conv` | [x] PASS (1.33->0.35) | [x] PASS (3.50->0.52) | [x] PASS (1.02->0.34) | [x] |  |
| `ECConv` | `k3_node.layers.conv.nn_conv` | [x] PASS (3.44->0.51) | [x] PASS (4.89->0.82) | [x] PASS (2.52->0.57) | [x] |  |
| `NNConv` | `k3_node.layers.conv.nn_conv` | [x] PASS (3.44->0.69) | [x] PASS (5.27->0.68) | [x] PASS (2.71->0.44) | [x] |  |
| `PANConv` | `k3_node.layers.conv.pan_conv` | [x] PASS (2.24->0.94) | [x] PASS (1.85->0.97) | [x] PASS (1.64->0.78) | [x] |  |
| `PDNConv` | `k3_node.layers.conv.pdn_conv` | [x] PASS (1.50->0.48) | [x] PASS (0.95->0.65) | [x] PASS (0.82->0.45) | [x] |  |
| `PNAConv` | `k3_node.layers.conv.pna_conv` | [x] PASS (4.39->0.63) | [x] PASS (5.79->0.69) | [x] PASS (11.61->1.00) | [x] |  |
|`PointConv`|`k3_node.layers.conv.point_conv`| [x] PASS (5.41->1.25) | [x] PASS (0.99->0.43) | [x] PASS (1.89->1.04) | [x] | |
|`PointNetConv`|`k3_node.layers.conv.point_conv`| [x] PASS (2.57->1.01) | [x] PASS (1.36->0.67) | [x] PASS (0.92->0.59) | [x] | |
|`PointGNNConv`|`k3_node.layers.conv.point_gnn_conv`| [x] PASS (3.49->0.82) | [x] PASS (7.33->1.41) | [x] PASS (5.33->1.01) | [x] | |
|`PointTransformerConv`|`k3_node.layers.conv.point_transformer_conv`| [x] PASS (2.46->0.89) | [x] PASS (2.04->0.77) | [x] PASS (3.23->0.82) | [x] | |
|`PPFConv`|`k3_node.layers.conv.ppf_conv`| [x] PASS (4.67->1.45) | [x] PASS (0.93->0.63) | [x] PASS (1.85->0.57) | [x] | |
| `PPNPPropagation` | `k3_node.layers.conv.ppnp` | [x] PASS (2.17->0.88) | [x] PASS (4.29->0.81) | [x] PASS (0.95->0.41) | [x] | |
|`ResGatedGraphConv`|`k3_node.layers.conv.res_gated_graph_conv`| [x] PASS (1.90->0.31) | [x] PASS (5.41->0.86) | [x] PASS (6.10->0.65) | [x] | |
|`RGATConv`|`k3_node.layers.conv.rgat_conv`| [x] PASS (0.96->0.44) | [x] PASS (0.94->0.39) | [x] PASS (0.91->0.36) | [x] | |
|`CuGraphRGCNConv`|`k3_node.layers.conv.rgcn_conv`| [-] SKIP | [-] SKIP | [-] SKIP | [-] | cuGraph vendor layer (requires GPU cuGraph bindings) |
|`FastRGCNConv`|`k3_node.layers.conv.rgcn_conv`| [x] PASS (2.50->0.52) | [x] PASS (3.63->0.36) | [x] PASS (1.69->0.33) | [x] | |
| `RGCNConv` | `k3_node.layers.conv.rgcn_conv` | [x] PASS (3.07->0.80) | [x] PASS (3.07->0.74) | [x] PASS (2.81->0.56) | [x] | |
|`SAGEConv`|`k3_node.layers.conv.sage_conv`| [x] PASS (3.76->0.66) | [x] PASS (3.53->0.89) | [x] PASS (3.11->0.91) | [x] | |
|`SGConv`|`k3_node.layers.conv.sg_conv`| [x] PASS (2.44->0.79) | [x] PASS (1.25->0.60) | [x] PASS (1.16->0.85) | [x] | |
|`SignedConv`|`k3_node.layers.conv.signed_conv`| [x] PASS (3.79->1.64) | [x] PASS (2.48->0.92) | [x] PASS (2.06->0.91) | [x] | |
|`SimpleConv`|`k3_node.layers.conv.simple_conv`| [x] PASS (3.58->0.69) | [x] PASS (3.35->0.77) | [x] PASS (8.31->1.13) | [x] | |
|`SplineConv`|`k3_node.layers.conv.spline_conv`| [x] PASS (2.72->0.87) | [x] PASS (1.70->0.57) | [x] PASS (1.86->0.59) | [x] | |
|`SSGConv`|`k3_node.layers.conv.ssg_conv`| [x] PASS (2.07->0.97) | [x] PASS (1.04->0.59) | [x] PASS (1.07->0.53) | [x] | |
|`SuperGATConv`|`k3_node.layers.conv.supergat_conv`| [x] PASS (1.46->0.84) | [x] PASS (0.95->0.62) | [x] PASS (1.72->0.69) | [x] | |
|`TAGConv`|`k3_node.layers.conv.tag_conv`| [x] PASS (4.53->0.56) | [x] PASS (3.60->0.50) | [x] PASS (3.27->0.87) | [x] | |
|`TransformerConv`|`k3_node.layers.conv.transformer_conv`| [x] PASS (2.93->0.79) | [x] PASS (1.23->0.34) | [x] PASS (3.29->0.57) | [x] | |
|`WLConv`|`k3_node.layers.conv.wl_conv`| [-] SKIP | [-] SKIP | [-] SKIP | [-] | Discrete non-differentiable Weisfeiler-Lehman graph coloring operator |
|`WLConvContinuous`|`k3_node.layers.conv.wl_conv`| [x] PASS (2.20->1.14) | [x] PASS (1.73->0.79) | [x] PASS (1.02->0.45) | [x] | |
|`XConv`|`k3_node.layers.conv.x_conv`| [x] PASS (1.22->0.50) | [x] PASS (0.84->0.60) | [x] PASS (0.89->0.40) | [x] | |

## Normalization Layers (`k3_node.layers.norm`)

| Layer Name | Module | PyTorch | TensorFlow | JAX | Learning Verified | Notes / Fixes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|`BatchNorm`|`k3_node.layers.norm.batch_norm`| [x] PASS (3.26->0.62) | [x] PASS (1.18->0.46) | [x] PASS (3.92->1.10) | [x] | |
|`HeteroBatchNorm`|`k3_node.layers.norm.batch_norm`| [-] SKIP | [-] SKIP | [-] SKIP | [-] | Heterogeneous Norm layer (tested in hetero suite) |
|`DiffGroupNorm`|`k3_node.layers.norm.diff_group_norm`| [x] PASS (2.75->0.74) | [x] PASS (1.53->0.35) | [x] PASS (2.23->0.73) | [x] | |
|`GraphNorm`|`k3_node.layers.norm.graph_norm`| [x] PASS (1.56->0.34) | [x] PASS (1.81->0.31) | [x] PASS (1.68->0.48) | [x] | |
|`GraphSizeNorm`|`k3_node.layers.norm.graph_size_norm`| [x] PASS (1.48->0.82) | [x] PASS (0.65->0.29) | [x] PASS (1.05->0.62) | [x] | |
|`InstanceNorm`|`k3_node.layers.norm.instance_norm`| [x] PASS (2.54->0.72) | [x] PASS (2.32->0.47) | [x] PASS (1.59->0.50) | [x] | |
|`HeteroLayerNorm`|`k3_node.layers.norm.layer_norm`| [-] SKIP | [-] SKIP | [-] SKIP | [-] | Heterogeneous Norm layer (tested in hetero suite) |
|`LayerNorm`|`k3_node.layers.norm.layer_norm`| [x] PASS (2.54->0.46) | [x] PASS (2.96->0.60) | [x] PASS (2.75->0.83) | [x] | |
|`MeanSubtractionNorm`|`k3_node.layers.norm.mean_subtraction_norm`| [x] PASS (2.92->0.78) | [x] PASS (1.51->0.60) | [x] PASS (2.31->0.44) | [x] | |
|`MessageNorm`|`k3_node.layers.norm.msg_norm`| [x] PASS (1.92->1.00) | [x] PASS (1.89->0.82) | [x] PASS (1.46->0.85) | [x] | |
|`PairNorm`|`k3_node.layers.norm.pair_norm`| [x] PASS (1.27->0.82) | [x] PASS (0.82->0.39) | [x] PASS (1.15->0.65) | [x] | |

## Pooling Layers (`k3_node.layers.pool`)

| Layer Name | Module | PyTorch | TensorFlow | JAX | Learning Verified | Notes / Fixes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|`ASAPooling`|`k3_node.layers.pool.asap`| [x] PASS (4.53->0.59) | [x] PASS (7.27->1.42) | [x] PASS (1.23->0.88) | [x] | |
|`ClusterPooling`|`k3_node.layers.pool.cluster_pool`| [x] PASS (10.19->1.31) | [x] PASS (7.35->0.79) | [x] PASS (29.41->3.73) | [x] | |
|`Connect`|`k3_node.layers.pool.connect.base`| [-] SKIP | [-] SKIP | [-] SKIP | [-] | Base/Abstract class |
|`FilterEdges`|`k3_node.layers.pool.connect.filter_edges`| [x] PASS (2.67->1.22) | [x] PASS (0.88->0.22) | [x] PASS (1.47->0.62) | [x] | |
|`EdgePooling`|`k3_node.layers.pool.edge_pool`| [x] PASS (1.24->0.13) | [x] PASS (9.27->1.04) | [x] PASS (14.50->2.56) | [x] | |
|`MemPooling`|`k3_node.layers.pool.mem_pool`| [x] PASS (5.27->0.67) | [x] PASS (47.20->5.85) | [x] PASS (8.39->0.66) | [x] | |
|`PANPooling`|`k3_node.layers.pool.pan_pool`| [x] PASS (3.15->0.22) | [x] PASS (0.80->0.17) | [x] PASS (1.77->0.58) | [x] | |
|`SAGPooling`|`k3_node.layers.pool.sag_pool`| [x] PASS (1.36->0.34) | [x] PASS (0.58->0.39) | [x] PASS (41.26->4.02) | [x] | |
|`Select`|`k3_node.layers.pool.select.base`| [-] SKIP | [-] SKIP | [-] SKIP | [-] | Base/Abstract class |
|`SelectTopK`|`k3_node.layers.pool.select.topk`| [x] PASS (0.17->0.01) | [x] PASS (1.52->0.87) | [x] PASS (0.09->0.01) | [x] | |
|`TopKPooling`|`k3_node.layers.pool.topk_pool`| [x] PASS (8.20->0.73) | [x] PASS (13.06->2.36) | [x] PASS (2.44->0.33) | [x] | |

## Aggregation Layers (`k3_node.layers.aggr`)

| Layer Name | Module | PyTorch | TensorFlow | JAX | Learning Verified | Notes / Fixes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|`AttentionalAggregation`|`k3_node.layers.aggr.attention`| [x] PASS (2.57->0.23) | [x] PASS (0.96->0.13) | [x] PASS (2.31->0.14) | [x] | |
|`Aggregation`|`k3_node.layers.aggr.base`| [-] SKIP | [-] SKIP | [-] SKIP | [-] | Base/Abstract class |
|`MaxAggregation`|`k3_node.layers.aggr.basic`| [x] PASS (1.71->0.35) | [x] PASS (2.95->0.27) | [x] PASS (0.61->0.04) | [x] | |
|`MeanAggregation`|`k3_node.layers.aggr.basic`| [x] PASS (1.95->0.51) | [x] PASS (0.89->0.04) | [x] PASS (0.75->0.12) | [x] | |
|`MinAggregation`|`k3_node.layers.aggr.basic`| [x] PASS (4.45->0.35) | [x] PASS (4.57->0.24) | [x] PASS (1.45->0.20) | [x] | |
|`MulAggregation`|`k3_node.layers.aggr.basic`| [x] PASS (13.81->0.50) | [x] PASS (2.92->0.20) | [x] PASS (3.08->0.19) | [x] | |
|`PowerMeanAggregation`|`k3_node.layers.aggr.basic`| [x] PASS (1.40->0.54) | [x] PASS (2.29->0.55) | [x] PASS (1.55->0.45) | [x] | |
|`SoftmaxAggregation`|`k3_node.layers.aggr.basic`| [x] PASS (1.79->0.25) | [x] PASS (4.42->0.29) | [x] PASS (2.12->0.14) | [x] | |
|`StdAggregation`|`k3_node.layers.aggr.basic`| [x] PASS (1.22->0.15) | [x] PASS (3.61->0.72) | [x] PASS (3.67->0.52) | [x] | |
|`SumAggregation`|`k3_node.layers.aggr.basic`| [x] PASS (6.28->0.46) | [x] PASS (4.35->0.37) | [x] PASS (10.11->0.89) | [x] | |
|`VarAggregation`|`k3_node.layers.aggr.basic`| [x] PASS (1.44->0.09) | [x] PASS (0.65->0.11) | [x] PASS (8.54->0.57) | [x] | |
|`DeepSetsAggregation`|`k3_node.layers.aggr.deep_sets`| [x] PASS (4.78->0.24) | [x] PASS (12.34->0.60) | [x] PASS (4.40->0.18) | [x] | |
|`EquilibriumAggregation`|`k3_node.layers.aggr.equilibrium`| [x] PASS (0.66->0.05) | [x] PASS (1.15->0.15) | [x] PASS (0.93->0.04) | [x] | |
|`FusedAggregation`|`k3_node.layers.aggr.fused`| [x] PASS (9.04->0.40) | [x] PASS (11.36->0.47) | [x] PASS (1.97->0.26) | [x] | |
|`GraphMultisetTransformer`|`k3_node.layers.aggr.gmt`| [x] PASS (1.46->0.26) | [x] PASS (10.83->0.12) | [x] PASS (1.08->0.25) | [x] | |
|`GRUAggregation`|`k3_node.layers.aggr.gru`| [x] PASS (1.44->0.51) | [x] PASS (0.65->0.27) | [x] PASS (0.41->0.12) | [x] | |
|`LCMAggregation`|`k3_node.layers.aggr.lcm`| [x] PASS (3.59->1.83) | [x] PASS (0.78->0.47) | [x] PASS (0.94->0.47) | [x] | |
|`LSTMAggregation`|`k3_node.layers.aggr.lstm`| [x] PASS (1.05->0.57) | [x] PASS (0.95->0.23) | [x] PASS (1.26->0.37) | [x] | |
|`MLPAggregation`|`k3_node.layers.aggr.mlp`| [x] PASS (1.23->0.12) | [x] PASS (1.68->0.11) | [x] PASS (1.76->0.10) | [x] | |
|`MultiAggregation`|`k3_node.layers.aggr.multi`| [x] PASS (6.64->0.71) | [x] PASS (3.51->0.40) | [x] PASS (2.01->0.14) | [x] | |
|`PatchTransformerAggregation`|`k3_node.layers.aggr.patch_transformer`| [x] PASS (2.77->0.31) | [x] PASS (1.18->0.09) | [x] PASS (0.75->0.03) | [x] | |
|`MedianAggregation`|`k3_node.layers.aggr.quantile`| [x] PASS (0.79->0.17) | [x] PASS (0.90->0.69) | [x] PASS (1.20->0.65) | [x] | |
|`QuantileAggregation`|`k3_node.layers.aggr.quantile`| [x] PASS (1.86->0.94) | [x] PASS (0.65->0.53) | [x] PASS (1.27->1.17) | [x] | |
|`DegreeScalerAggregation`|`k3_node.layers.aggr.scaler`| [x] PASS (6.35->0.70) | [x] PASS (2.36->0.16) | [x] PASS (5.46->0.54) | [x] | |
|`Set2Set`|`k3_node.layers.aggr.set2set`| [x] PASS (1.24->0.10) | [x] PASS (1.72->0.08) | [x] PASS (0.46->0.03) | [x] | |
|`SetTransformerAggregation`|`k3_node.layers.aggr.set_transformer`| [x] PASS (4.50->0.26) | [x] PASS (3.58->0.19) | [x] PASS (2.13->0.12) | [x] | |
|`SortAggregation`|`k3_node.layers.aggr.sort`| [x] PASS (1.75->0.05) | [x] PASS (3.30->0.39) | [x] PASS (4.29->0.31) | [x] | |
|`VariancePreservingAggregation`|`k3_node.layers.aggr.variance_preserving`| [x] PASS (4.18->0.20) | [x] PASS (1.65->0.15) | [x] PASS (3.98->0.25) | [x] | |

## Dense / Linear Layers (`k3_node.layers.dense`)

| Layer Name | Module | PyTorch | TensorFlow | JAX | Learning Verified | Notes / Fixes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|`DenseGATConv`|`k3_node.layers.dense.dense_gat_conv`| [x] PASS (2.13->1.07) | [x] PASS (1.51->0.80) | [x] PASS (1.96->1.10) | [x] | |
|`DenseGCNConv`|`k3_node.layers.dense.dense_gcn_conv`| [x] PASS (2.32->1.14) | [x] PASS (3.61->1.76) | [x] PASS (1.53->0.73) | [x] | |
|`DenseGINConv`|`k3_node.layers.dense.dense_gin_conv`| [x] PASS (8.92->3.54) | [x] PASS (12.85->5.62) | [x] PASS (5.16->1.93) | [x] | |
|`DenseGraphConv`|`k3_node.layers.dense.dense_graph_conv`| [x] PASS (0.92->0.22) | [x] PASS (1.64->0.66) | [x] PASS (1.67->0.70) | [x] | |
|`DenseSAGEConv`|`k3_node.layers.dense.dense_sage_conv`| [x] PASS (1.88->0.59) | [x] PASS (1.23->0.47) | [x] PASS (2.96->0.97) | [x] | |
| `DMoNPooling` | `k3_node.layers.dense.dmon_pool` | [x] PASS (3.01->0.30) | [x] PASS (0.65->0.02) | [x] PASS (1.22->0.02) | [x] | |
|`HeteroDictLinear`|`k3_node.layers.dense.linear`| [-] SKIP | [-] SKIP | [-] SKIP | [-] | Heterogeneous Linear layer (tested in hetero suite) |
|`HeteroLinear`|`k3_node.layers.dense.linear`| [-] SKIP | [-] SKIP | [-] SKIP | [-] | Heterogeneous Linear layer (tested in hetero suite) |
|`Linear`|`k3_node.layers.dense.linear`| [x] PASS (1.05->0.57) | [x] PASS (0.95->0.46) | [x] PASS (1.38->0.64) | [x] | |

## Attention Layers (`k3_node.layers.attention`)

| Layer Name | Module | PyTorch | TensorFlow | JAX | Learning Verified | Notes / Fixes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|`PerformerAttention`|`k3_node.layers.attention.performer`| [x] PASS (0.75->0.32) | [x] PASS (1.36->0.57) | [x] PASS (0.84->0.23) | [x] | |
| `PerformerProjection` | `k3_node.layers.attention.performer` | [x] PASS (1.43->0.92) | [x] PASS (2.24->1.39) | [x] PASS (1.52->1.14) | [x] | |
|`PolynormerAttention`|`k3_node.layers.attention.polynormer`| [x] PASS (3.23->0.51) | [x] PASS (1.89->0.47) | [x] PASS (1.12->0.43) | [x] | |
|`QFormer`|`k3_node.layers.attention.qformer`| [x] PASS (2.33->0.74) | [x] PASS (2.01->0.30) | [x] PASS (1.70->0.73) | [x] | |
| `QFormerEncoderLayer` | `k3_node.layers.attention.qformer` | [x] PASS (2.12->0.39) | [x] PASS (1.77->0.27) | [x] PASS (2.25->0.45) | [x] | |
| `SGFormerAttention` | `k3_node.layers.attention.sgformer` | [x] PASS (1.28->0.20) | [x] PASS (1.02->0.12) | [x] PASS (0.82->0.17) | [x] | |

## Knowledge Graph Embedding (`k3_node.layers.kge`)

| Layer Name | Module | PyTorch | TensorFlow | JAX | Learning Verified | Notes / Fixes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|`KGEModel`|`k3_node.layers.kge.base`| [-] SKIP | [-] SKIP | [-] SKIP | [-] | Base/Abstract class |
|`ComplEx`|`k3_node.layers.kge.complex`| [x] PASS (1.02->0.18) | [x] PASS (1.09->0.12) | [x] PASS (1.07->0.16) | [x] | |
|`DistMult`|`k3_node.layers.kge.distmult`| [x] PASS (1.11->0.64) | [x] PASS (1.01->0.49) | [x] PASS (1.05->0.72) | [x] | |
|`RotatE`|`k3_node.layers.kge.rotate`| [x] PASS (2.46->0.44) | [x] PASS (2.10->0.27) | [x] PASS (2.08->0.34) | [x] | |
|`TransE`|`k3_node.layers.kge.transe`| [x] PASS (7.33->4.26) | [x] PASS (7.90->3.67) | [x] PASS (7.30->4.67) | [x] | |

