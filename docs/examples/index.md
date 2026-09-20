# Code Examples

Welcome to the **K3-Node Code Examples**. This page indexes end-to-end runnable tutorials demonstrating how to build, train, and evaluate Graph Neural Networks with K3-Node across multiple frameworks and backends.

---

## Available Examples

<div class="grid cards" markdown>

-   :material-graph: __Attention-based Graph Neural Network (AGNN) on Cora__

    ---

    Node classification using AGNNConv with dynamic attention-based propagation weights.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `AGNNConv`

    [:octicons-arrow-right-24: Read Tutorial](agnn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/agnn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/agnn.ipynb){ .md-button }

-   :material-vector-link: __Attract-Repel Link Prediction on Cora__

    ---

    Link prediction with Attract-Repel loss enforcing neighborhood affinity and negative repulsion.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `ARLinkPredictor`

    [:octicons-arrow-right-24: Read Tutorial](ar_link_pred.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/ar_link_pred.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/ar_link_pred.ipynb){ .md-button }

-   :material-shield-sync: __Adversarially Regularized Variational Graph Autoencoder (ARGVA)__

    ---

    Graph representation learning and clustering via adversarial variational autoencoding.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `ARGVA`

    [:octicons-arrow-right-24: Read Tutorial](argva_node_clustering.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/argva_node_clustering.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/argva_node_clustering.ipynb){ .md-button }

-   :material-chart-bell-curve: __Auto-Regressive Moving Average Graph Convolution (ARMAConv) on Cora__

    ---

    Node classification using ARMA filters for localized, multi-scale neighborhood aggregation.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `ARMAConv`

    [:octicons-arrow-right-24: Read Tutorial](arma.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/arma.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/arma.ipynb){ .md-button }

-   :material-molecule: __AttentiveFP Molecular Property Prediction__

    ---

    Molecular property prediction with Attentive Fingerprint graph neural network.

    - **Backend**: Multi-Backend
    - **Dataset**: `MoleculeNet`
    - **Key Layer**: `AttentiveFP`

    [:octicons-arrow-right-24: Read Tutorial](attentive_fp.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/attentive_fp.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/attentive_fp.ipynb){ .md-button }

-   :material-vector-combine: __Graph Autoencoders (GAE & VGAE) on Cora__

    ---

    Unsupervised graph representation learning and link prediction with GAE and VGAE.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `GAE / VGAE`

    [:octicons-arrow-right-24: Read Tutorial](autoencoder.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/autoencoder.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/autoencoder.ipynb){ .md-button }

-   :material-server-network: __Cluster-GCN on PPI Graph Dataset__

    ---

    Scalable training via graph partitioning (Cluster-GCN) on the PPI dataset.

    - **Backend**: Multi-Backend
    - **Dataset**: `PPI`
    - **Key Layer**: `SAGEConv`

    [:octicons-arrow-right-24: Read Tutorial](cluster_gcn_ppi.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/cluster_gcn_ppi.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/cluster_gcn_ppi.ipynb){ .md-button }

-   :material-reddit: __Cluster-GCN on Reddit Graph__

    ---

    Training large-scale GCN on Reddit by partitioning nodes into subgraphs.

    - **Backend**: Multi-Backend
    - **Dataset**: `Reddit`
    - **Key Layer**: `SAGEConv`

    [:octicons-arrow-right-24: Read Tutorial](cluster_gcn_reddit.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/cluster_gcn_reddit.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/cluster_gcn_reddit.ipynb){ .md-button }

-   :material-palette-outline: __Graph Classification with TopKPooling on Colors Dataset__

    ---

    Hierarchical graph representation learning using TopKPooling.

    - **Backend**: Multi-Backend
    - **Dataset**: `TUDataset (Colors)`
    - **Key Layer**: `TopKPooling`

    [:octicons-arrow-right-24: Read Tutorial](colors_topk_pool.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/colors_topk_pool.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/colors_topk_pool.ipynb){ .md-button }

-   :material-book-open-variant: __Node Classification on Cora Benchmark__

    ---

    Benchmark comparison for semi-supervised node classification on Cora.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `SplineConv / GCNConv`

    [:octicons-arrow-right-24: Read Tutorial](cora.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/cora.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/cora.ipynb){ .md-button }

-   :material-auto-fix: __Correct and Smooth (C&S) Post-Processing on Cora__

    ---

    Combining simple base MLP predictions with graph error-correction and smoothing.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `CorrectAndSmooth`

    [:octicons-arrow-right-24: Read Tutorial](correct_and_smooth.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/correct_and_smooth.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/correct_and_smooth.ipynb){ .md-button }

-   :material-pipe: __PyG Graph DataPipe & Streaming Loaders__

    ---

    Streaming graph data loading and iterative data pipelines.

    - **Backend**: Multi-Backend
    - **Dataset**: `Synthetic Meshes`
    - **Key Layer**: `DataLoader`

    [:octicons-arrow-right-24: Read Tutorial](datapipe.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/datapipe.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/datapipe.ipynb){ .md-button }

-   :material-axis-arrow: __Dynamic Graph CNN (DGCNN) for Point Cloud Classification__

    ---

    3D point cloud classification with dynamic k-NN graphs and EdgeConv.

    - **Backend**: Multi-Backend
    - **Dataset**: `ModelNet / MedShapeNet`
    - **Key Layer**: `DynamicEdgeConv`

    [:octicons-arrow-right-24: Read Tutorial](dgcnn_classification.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/dgcnn_classification.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/dgcnn_classification.ipynb){ .md-button }

-   :material-cube-scan: __Dynamic Graph CNN for 3D Part Segmentation__

    ---

    Part-level 3D point cloud segmentation with dynamic edge convolutions.

    - **Backend**: Multi-Backend
    - **Dataset**: `ShapeNet`
    - **Key Layer**: `DynamicEdgeConv`

    [:octicons-arrow-right-24: Read Tutorial](dgcnn_segmentation.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/dgcnn_segmentation.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/dgcnn_segmentation.ipynb){ .md-button }

-   :material-arrow-decision: __Directed Graph Neural Network (DirGNN) on Web Graphs__

    ---

    Convolution over directed graphs disentangling incoming and outgoing edge information.

    - **Backend**: Multi-Backend
    - **Dataset**: `WikipediaNetwork`
    - **Key Layer**: `DirGNNConv`

    [:octicons-arrow-right-24: Read Tutorial](dir_gnn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/dir_gnn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/dir_gnn.ipynb){ .md-button }

-   :material-dna: __Dynamic Neighborhood Aggregation (DNAConv) on Cora__

    ---

    Deep GNNs with multi-head dynamic neighborhood aggregation and attention.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `DNAConv`

    [:octicons-arrow-right-24: Read Tutorial](dna.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/dna.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/dna.ipynb){ .md-button }

-   :material-lightning-bolt: __Efficient Graph Convolution (EGConv) on Cora__

    ---

    Multi-head and multi-scale efficient graph convolution.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `EGConv`

    [:octicons-arrow-right-24: Read Tutorial](egc.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/egc.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/egc.ipynb){ .md-button }

-   :material-scale-balance: __Equilibrium Aggregation on Graph Benchmarks__

    ---

    Robust graph representation learning using equilibrium-based median aggregation.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora`
    - **Key Layer**: `EquilibriumAggregation`

    [:octicons-arrow-right-24: Read Tutorial](equilibrium_median.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/equilibrium_median.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/equilibrium_median.ipynb){ .md-button }

-   :material-vector-polygon: __SplineConv on FAUST 3D Mesh Registration__

    ---

    Continuous B-spline convolutions on non-Euclidean 3D mesh surfaces.

    - **Backend**: Multi-Backend
    - **Dataset**: `FAUST`
    - **Key Layer**: `SplineConv`

    [:octicons-arrow-right-24: Read Tutorial](faust.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/faust.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/faust.ipynb){ .md-button }

-   :material-movie-open: __Feature-wise Linear Modulation (FiLMConv) on Cora__

    ---

    Hypernetwork-driven feature-wise modulation of neighbor message passing.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `FiLMConv`

    [:octicons-arrow-right-24: Read Tutorial](film.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/film.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/film.ipynb){ .md-button }

-   :material-eye: __Graph Attention Networks (GAT & GATv2) on Cora__

    ---

    Multi-head attention mechanisms assigning dynamic importance weights to graph edges.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `GATConv / GATv2Conv`

    [:octicons-arrow-right-24: Read Tutorial](gat.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/gat.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/gat.ipynb){ .md-button }

-   :material-graphql: __Graph Convolutional Network (GCN) on Cora__

    ---

    Canonical semi-supervised node classification on citation networks.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `GCNConv`

    [:octicons-arrow-right-24: Read Tutorial](gcn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/gcn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/gcn.ipynb){ .md-button }

-   :material-layers-triple: __Deep GCN with Initial Residual Connections (GCNII) on Cora__

    ---

    Deep GNN architecture resolving over-smoothing via initial residuals and identity mapping.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `GCN2Conv`

    [:octicons-arrow-right-24: Read Tutorial](gcn2_cora.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/gcn2_cora.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/gcn2_cora.ipynb){ .md-button }

-   :material-graph-outline: __GCNII on Protein-Protein Interaction (PPI) Dataset__

    ---

    Deep multi-layer GCNII for inductive multi-label protein interaction prediction.

    - **Backend**: Multi-Backend
    - **Dataset**: `PPI`
    - **Key Layer**: `GCN2Conv`

    [:octicons-arrow-right-24: Read Tutorial](gcn2_ppi.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/gcn2_ppi.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/gcn2_ppi.ipynb){ .md-button }

-   :material-magic-staff: __Adaptive Neighborhood Exploration with GeniePath__

    ---

    Gated path-based neighborhood exploration using LSTM memory cells.

    - **Backend**: Multi-Backend
    - **Dataset**: `PPI / Cora`
    - **Key Layer**: `GeniePathConv`

    [:octicons-arrow-right-24: Read Tutorial](geniepath.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/geniepath.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/geniepath.ipynb){ .md-button }

-   :material-school: __Graph-to-MLP Knowledge Distillation (GLNN)__

    ---

    Distilling relational knowledge from teacher GNNs into inference-efficient student MLPs.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `GCN / MLP`

    [:octicons-arrow-right-24: Read Tutorial](glnn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/glnn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/glnn.ipynb){ .md-button }

-   :material-compass: __Graph Positional and Structural Embeddings (GPSE)__

    ---

    Learning rich positional and structural node features for expressive GNNs.

    - **Backend**: Multi-Backend
    - **Dataset**: `ZINC`
    - **Key Layer**: `GPSE`

    [:octicons-arrow-right-24: Read Tutorial](gpse.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/gpse.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/gpse.ipynb){ .md-button }

-   :material-satellite-variant: __General Powerful Scalable Graph Transformer (GPS)__

    ---

    Hybrid architecture combining local message passing with global full-attention transformers.

    - **Backend**: Multi-Backend
    - **Dataset**: `ZINC`
    - **Key Layer**: `GPSLayer`

    [:octicons-arrow-right-24: Read Tutorial](graph_gps.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graph_gps.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graph_gps.ipynb){ .md-button }

-   :material-brain: __Unsupervised GraphSAGE on Citation Network__

    ---

    Inductive representation learning via negative-sampling random-walk objectives.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora / KarateClub`
    - **Key Layer**: `SAGEConv`

    [:octicons-arrow-right-24: Read Tutorial](graph_sage_unsup.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graph_sage_unsup.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graph_sage_unsup.ipynb){ .md-button }

-   :material-circle-multiple-outline: __Unsupervised GraphSAGE on PPI Dataset__

    ---

    Inductive unsupervised node embeddings on multi-graph protein interactions.

    - **Backend**: Multi-Backend
    - **Dataset**: `PPI`
    - **Key Layer**: `SAGEConv`

    [:octicons-arrow-right-24: Read Tutorial](graph_sage_unsup_ppi.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graph_sage_unsup_ppi.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graph_sage_unsup_ppi.ipynb){ .md-button }

-   :material-chart-scatter-plot: __GraphSAINT Inductive Subgraph Sampling on Reddit__

    ---

    Scalable training via random-walk and edge-sampling subgraph extraction.

    - **Backend**: Multi-Backend
    - **Dataset**: `Reddit`
    - **Key Layer**: `GraphSAINT / SAGEConv`

    [:octicons-arrow-right-24: Read Tutorial](graph_saint.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graph_saint.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graph_saint.ipynb){ .md-button }

-   :material-tune: __Graph U-Net with gPool and gUnpool__

    ---

    Encoder-decoder graph architecture with top-k node downsampling and upsampling.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `GraphUNet / TopKPooling`

    [:octicons-arrow-right-24: Read Tutorial](graph_unet.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graph_unet.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graph_unet.ipynb){ .md-button }

-   :material-terrain: __GraphLand Benchmark Pipeline__

    ---

    Benchmarking message passing architectures across diverse graph topologies.

    - **Backend**: Multi-Backend
    - **Dataset**: `GraphLand`
    - **Key Layer**: `GCNConv`

    [:octicons-arrow-right-24: Read Tutorial](graphland.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graphland.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graphland.ipynb){ .md-button }

-   :material-family-tree: __Hierarchical Neighborhood Sampling__

    ---

    Layer-wise hierarchical mini-batch sampling for large graphs.

    - **Backend**: Multi-Backend
    - **Dataset**: `Reddit / Flickr`
    - **Key Layer**: `NeighborLoader`

    [:octicons-arrow-right-24: Read Tutorial](hierarchical_sampling.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/hierarchical_sampling.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/hierarchical_sampling.ipynb){ .md-button }

-   :material-information-outline: __Deep Graph Infomax (Inductive PPI)__

    ---

    Maximizing mutual information between local node patches and global graph summary.

    - **Backend**: Multi-Backend
    - **Dataset**: `PPI`
    - **Key Layer**: `DeepGraphInfomax`

    [:octicons-arrow-right-24: Read Tutorial](infomax_inductive.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/infomax_inductive.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/infomax_inductive.ipynb){ .md-button }

-   :material-information: __Deep Graph Infomax (Transductive Cora)__

    ---

    Unsupervised node embeddings by contrasting local vs corrupted global graph representations.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `DeepGraphInfomax`

    [:octicons-arrow-right-24: Read Tutorial](infomax_transductive.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/infomax_transductive.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/infomax_transductive.ipynb){ .md-button }

-   :material-database-search: __Knowledge Graph Embeddings (TransE / DistMult) on FB15k-237__

    ---

    Translational and bilinear knowledge graph completion on FB15k-237.

    - **Backend**: Multi-Backend
    - **Dataset**: `FB15k-237`
    - **Key Layer**: `TransE / DistMult`

    [:octicons-arrow-right-24: Read Tutorial](kge_fb15k_237.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/kge_fb15k_237.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/kge_fb15k_237.ipynb){ .md-button }

-   :material-broadcast: __Label Propagation Algorithm (LPA) on Cora__

    ---

    Iterative diffusion of known labels over graph edges without learnable parameters.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `LabelPropagation`

    [:octicons-arrow-right-24: Read Tutorial](label_prop.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/label_prop.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/label_prop.ipynb){ .md-button }

-   :material-function-variant: __Custom Aggregation Operators: Second-Min & Order Statistics__

    ---

    Customizable generalized aggregation functions in graph message passing.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora`
    - **Key Layer**: `Aggregation`

    [:octicons-arrow-right-24: Read Tutorial](lcm_aggr_2nd_min.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/lcm_aggr_2nd_min.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/lcm_aggr_2nd_min.ipynb){ .md-button }

-   :material-star-outline: __LightGCN Recommender System on MovieLens__

    ---

    Simplified linear neighborhood aggregation for bipartite user-item recommendation.

    - **Backend**: Multi-Backend
    - **Dataset**: `MovieLens`
    - **Key Layer**: `LightGCN / BPRLoss`

    [:octicons-arrow-right-24: Read Tutorial](lightgcn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/lightgcn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/lightgcn.ipynb){ .md-button }

-   :material-link-variant-plus: __Link Prediction with GCN on Cora__

    ---

    Predicting edge existence via node embeddings and dot-product decoders.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `GCNConv`

    [:octicons-arrow-right-24: Read Tutorial](link_pred.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/link_pred.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/link_pred.ipynb){ .md-button }

-   :material-swap-horizontal-bold: __LINKX on Large Heterophilous Citation Graphs__

    ---

    Decoupled feature and structure transformations tailored for heterophilous graphs.

    - **Backend**: Multi-Backend
    - **Dataset**: `Penn94 / Cora`
    - **Key Layer**: `LINKX`

    [:octicons-arrow-right-24: Read Tutorial](linkx.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/linkx.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/linkx.ipynb){ .md-button }

-   :material-cpu-64-bit: __LPFormer: Link Prediction with Transformers__

    ---

    Transformer-based relational attention for link prediction.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora`
    - **Key Layer**: `LPFormer`

    [:octicons-arrow-right-24: Read Tutorial](lpformer.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/lpformer.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/lpformer.ipynb){ .md-button }

-   :material-memory: __Memory-Based Graph Pooling (MemPooling) on MUTAG__

    ---

    Clustering graph nodes using key-value memory addressing.

    - **Backend**: Multi-Backend
    - **Dataset**: `MUTAG (TUDataset)`
    - **Key Layer**: `MemPooling`

    [:octicons-arrow-right-24: Read Tutorial](mem_pool.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mem_pool.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mem_pool.ipynb){ .md-button }

-   :material-hopscotch: __MixHop: Higher-Order Neighborhood Convolution on Cora__

    ---

    Simultaneous mixing of 0-hop, 1-hop, and multi-hop neighborhood features.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `MixHopConv`

    [:octicons-arrow-right-24: Read Tutorial](mixhop.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mixhop.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mixhop.ipynb){ .md-button }

-   :material-image-size-select-actual: __Superpixel MNIST Classification with Graclus Pooling__

    ---

    Image classification on irregular superpixel graphs with Graclus coarsening.

    - **Backend**: Multi-Backend
    - **Dataset**: `MNISTSuperpixels`
    - **Key Layer**: `SplineConv / Graclus`

    [:octicons-arrow-right-24: Read Tutorial](mnist_graclus.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mnist_graclus.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mnist_graclus.ipynb){ .md-button }

-   :material-matrix: __Superpixel MNIST with Continuous Edge Convolutions (NNConv)__

    ---

    Dynamic continuous filter weights conditioned on relative spatial coordinates.

    - **Backend**: Multi-Backend
    - **Dataset**: `MNISTSuperpixels`
    - **Key Layer**: `NNConv`

    [:octicons-arrow-right-24: Read Tutorial](mnist_nn_conv.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mnist_nn_conv.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mnist_nn_conv.ipynb){ .md-button }

-   :material-grid: __Superpixel MNIST with Voxel Grid Downsampling__

    ---

    Voxel-grid geometric downsampling and graph pooling on superpixels.

    - **Backend**: Multi-Backend
    - **Dataset**: `MNISTSuperpixels`
    - **Key Layer**: `VoxelGrid / SplineConv`

    [:octicons-arrow-right-24: Read Tutorial](mnist_voxel_grid.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mnist_voxel_grid.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mnist_voxel_grid.ipynb){ .md-button }

-   :material-chemical-weapon: __Graph Isomorphism Network (GIN) on MUTAG__

    ---

    Maximally expressive Weisfeiler-Lehman graph classification on MUTAG molecules.

    - **Backend**: Multi-Backend
    - **Dataset**: `MUTAG (TUDataset)`
    - **Key Layer**: `GINConv`

    [:octicons-arrow-right-24: Read Tutorial](mutag_gin.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mutag_gin.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mutag_gin.ipynb){ .md-button }

-   :material-vector-square: __Node2Vec Representation Learning on KarateClub__

    ---

    Biased second-order random walks balancing breadth-first and depth-first exploration.

    - **Backend**: Multi-Backend
    - **Dataset**: `KarateClub`
    - **Key Layer**: `Node2Vec`

    [:octicons-arrow-right-24: Read Tutorial](node2vec.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/node2vec.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/node2vec.ipynb){ .md-button }

-   :material-dna: __DeepGCN with Residual Connections on OGBN-Proteins__

    ---

    Very deep GNNs (up to 28+ layers) with residual and dense skip connections.

    - **Backend**: Multi-Backend
    - **Dataset**: `ogbn-proteins`
    - **Key Layer**: `DeepGCNLayer`

    [:octicons-arrow-right-24: Read Tutorial](ogbn_proteins_deepgcn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/ogbn_proteins_deepgcn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/ogbn_proteins_deepgcn.ipynb){ .md-button }

-   :material-timer-sand: __OGBN Graph Benchmark Training Pipeline__

    ---

    Standardized scalable training pipeline on Open Graph Benchmark datasets.

    - **Backend**: Multi-Backend
    - **Dataset**: `ogbn-arxiv`
    - **Key Layer**: `SAGEConv / GCNConv`

    [:octicons-arrow-right-24: Read Tutorial](ogbn_train.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/ogbn_train.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/ogbn_train.ipynb){ .md-button }

-   :material-circle-slice-8: __Online Graph Clustering (OGC) with GNNs__

    ---

    Streaming and online graph clustering using dynamic neighborhood updates.

    - **Backend**: Multi-Backend
    - **Dataset**: `Reddit`
    - **Key Layer**: `ClusterGCNConv`

    [:octicons-arrow-right-24: Read Tutorial](ogc.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/ogc.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/ogc.ipynb){ .md-button }

-   :material-fast-forward: __Propagational MLP (PMLP) on Cora__

    ---

    Ultra-fast graph inference decoupling feature learning from message propagation.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `PMLP`

    [:octicons-arrow-right-24: Read Tutorial](pmlp.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/pmlp.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/pmlp.ipynb){ .md-button }

-   :material-variable: __Principal Neighbourhood Aggregation (PNA) on ZINC__

    ---

    Multiple statistical aggregators combined with degree-scaled amplification.

    - **Backend**: Multi-Backend
    - **Dataset**: `ZINC`
    - **Key Layer**: `PNAConv`

    [:octicons-arrow-right-24: Read Tutorial](pna.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/pna.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/pna.ipynb){ .md-button }

-   :material-shape-plus: __Point Transformer for 3D Shape Classification__

    ---

    Vector self-attention transformer for 3D geometric point set classification.

    - **Backend**: Multi-Backend
    - **Dataset**: `ModelNet40`
    - **Key Layer**: `PointTransformerConv`

    [:octicons-arrow-right-24: Read Tutorial](point_transformer_classification.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/point_transformer_classification.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/point_transformer_classification.ipynb){ .md-button }

-   :material-puzzle-outline: __Point Transformer for 3D Part Segmentation__

    ---

    Point transformer architecture for dense semantic 3D part labeling.

    - **Backend**: Multi-Backend
    - **Dataset**: `ShapeNet`
    - **Key Layer**: `PointTransformerConv`

    [:octicons-arrow-right-24: Read Tutorial](point_transformer_segmentation.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/point_transformer_segmentation.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/point_transformer_segmentation.ipynb){ .md-button }

-   :material-cube-outline: __PointNet++ for 3D Point Cloud Classification__

    ---

    Hierarchical neural network applying PointNet recursively on nested point partitions.

    - **Backend**: Multi-Backend
    - **Dataset**: `ModelNet40`
    - **Key Layer**: `PointNetConv`

    [:octicons-arrow-right-24: Read Tutorial](pointnet2_classification.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/pointnet2_classification.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/pointnet2_classification.ipynb){ .md-button }

-   :material-toy-brick-outline: __PointNet++ for 3D Point Cloud Segmentation__

    ---

    Hierarchical feature propagation and interpolation for dense 3D point cloud segmentation.

    - **Backend**: Multi-Backend
    - **Dataset**: `ShapeNet`
    - **Key Layer**: `PointNetConv`

    [:octicons-arrow-right-24: Read Tutorial](pointnet2_segmentation.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/pointnet2_segmentation.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/pointnet2_segmentation.ipynb){ .md-button }

-   :material-shield-account-outline: __Inductive Multi-Label Classification with GAT on PPI__

    ---

    Predicting protein functions across unseen graphs with Graph Attention Networks.

    - **Backend**: Multi-Backend
    - **Dataset**: `PPI`
    - **Key Layer**: `GATConv`

    [:octicons-arrow-right-24: Read Tutorial](ppi.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/ppi.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/ppi.ipynb){ .md-button }

-   :material-layers-plus: __Hierarchical Graph Representation with DiffPool on PROTEINS__

    ---

    Differentiable graph pooling learning hierarchical cluster assignment matrices.

    - **Backend**: Multi-Backend
    - **Dataset**: `PROTEINS (TUDataset)`
    - **Key Layer**: `DenseDiffPool`

    [:octicons-arrow-right-24: Read Tutorial](proteins_diff_pool.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/proteins_diff_pool.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/proteins_diff_pool.ipynb){ .md-button }

-   :material-view-grid-plus: __Deep Modular Network (DMoN) Pooling on PROTEINS__

    ---

    Graph pooling inspired by Newman modularity maximization.

    - **Backend**: Multi-Backend
    - **Dataset**: `PROTEINS (TUDataset)`
    - **Key Layer**: `DMoNPooling`

    [:octicons-arrow-right-24: Read Tutorial](proteins_dmon_pool.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/proteins_dmon_pool.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/proteins_dmon_pool.ipynb){ .md-button }

-   :material-set-all: __Graph Multiset Transformer (GMT) Pooling on PROTEINS__

    ---

    Multi-head attention pooling capturing inter-node interactions across whole graphs.

    - **Backend**: Multi-Backend
    - **Dataset**: `PROTEINS (TUDataset)`
    - **Key Layer**: `GraphMultisetTransformer`

    [:octicons-arrow-right-24: Read Tutorial](proteins_gmt.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/proteins_gmt.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/proteins_gmt.ipynb){ .md-button }

-   :material-content-cut: __MinCutPooling on PROTEINS Graph Benchmark__

    ---

    Spectral graph clustering formulated as a differentiable continuous min-cut objective.

    - **Backend**: Multi-Backend
    - **Dataset**: `PROTEINS (TUDataset)`
    - **Key Layer**: `MinCutPooling`

    [:octicons-arrow-right-24: Read Tutorial](proteins_mincut_pool.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/proteins_mincut_pool.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/proteins_mincut_pool.ipynb){ .md-button }

-   :material-filter-variant: __TopKPooling on PROTEINS Graph Classification__

    ---

    Hierarchical sparse node selection based on projection onto a learnable score vector.

    - **Backend**: Multi-Backend
    - **Dataset**: `PROTEINS (TUDataset)`
    - **Key Layer**: `TopKPooling`

    [:octicons-arrow-right-24: Read Tutorial](proteins_topk_pool.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/proteins_topk_pool.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/proteins_topk_pool.ipynb){ .md-button }

-   :material-atom-variant: __Molecular Property Prediction on QM9 with NNConv__

    ---

    Continuous edge-conditioned convolutions predicting quantum chemical properties.

    - **Backend**: Multi-Backend
    - **Dataset**: `QM9`
    - **Key Layer**: `NNConv`

    [:octicons-arrow-right-24: Read Tutorial](qm9_nn_conv.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/qm9_nn_conv.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/qm9_nn_conv.ipynb){ .md-button }

-   :material-atom: __Pretrained DimeNet++ on QM9 Molecular Properties__

    ---

    Directional message passing incorporating bond angles and interatomic distances.

    - **Backend**: Multi-Backend
    - **Dataset**: `QM9`
    - **Key Layer**: `DimeNetPlusPlus`

    [:octicons-arrow-right-24: Read Tutorial](qm9_pretrained_dimenet.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/qm9_pretrained_dimenet.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/qm9_pretrained_dimenet.ipynb){ .md-button }

-   :material-radioactive-circle-outline: __Pretrained SchNet on QM9 Molecular Benchmarks__

    ---

    Continuous-filter convolutional network for modeling quantum chemical interactions.

    - **Backend**: Multi-Backend
    - **Dataset**: `QM9`
    - **Key Layer**: `SchNet`

    [:octicons-arrow-right-24: Read Tutorial](qm9_pretrained_schnet.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/qm9_pretrained_schnet.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/qm9_pretrained_schnet.ipynb){ .md-button }

-   :material-radar: __RandLA-Net for Large-Scale Point Cloud Classification__

    ---

    Efficient point cloud architecture using random point sampling and local feature aggregation.

    - **Backend**: Multi-Backend
    - **Dataset**: `ModelNet40`
    - **Key Layer**: `RandLANet`

    [:octicons-arrow-right-24: Read Tutorial](randlanet_classification.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/randlanet_classification.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/randlanet_classification.ipynb){ .md-button }

-   :material-chart-scatter-plot-hexbin: __RandLA-Net for Semantic Point Cloud Segmentation__

    ---

    Real-time semantic segmentation on million-scale 3D point cloud scans.

    - **Backend**: Multi-Backend
    - **Dataset**: `S3DIS / ShapeNet`
    - **Key Layer**: `RandLANet`

    [:octicons-arrow-right-24: Read Tutorial](randlanet_segmentation.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/randlanet_segmentation.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/randlanet_segmentation.ipynb){ .md-button }

-   :material-database-sync: __Relational Deep Learning (RDL) Pipeline__

    ---

    Deep learning across multi-table relational databases via heterogeneous graph modeling.

    - **Backend**: Multi-Backend
    - **Dataset**: `Relational Databases`
    - **Key Layer**: `HeteroConv`

    [:octicons-arrow-right-24: Read Tutorial](rdl.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rdl.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rdl.ipynb){ .md-button }

-   :material-vector-selection: __RECT: Convex Hull Node Representation Learning__

    ---

    Convex hull objective learning robust node embeddings for graphs with extreme class imbalance.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora / WikipediaNetwork`
    - **Key Layer**: `RECT_L`

    [:octicons-arrow-right-24: Read Tutorial](rect.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rect.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rect.ipynb){ .md-button }

-   :material-reddit: __Inductive Node Classification on Reddit with SAGEConv__

    ---

    GraphSAGE training on the large Reddit community interaction graph.

    - **Backend**: Multi-Backend
    - **Dataset**: `Reddit`
    - **Key Layer**: `SAGEConv`

    [:octicons-arrow-right-24: Read Tutorial](reddit.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/reddit.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/reddit.ipynb){ .md-button }

-   :material-table-sync: __RelBench Relational Benchmark with GNNs__

    ---

    End-to-end relational table learning using multi-relational graph convolutions.

    - **Backend**: Multi-Backend
    - **Dataset**: `RelBench`
    - **Key Layer**: `HeteroGNN`

    [:octicons-arrow-right-24: Read Tutorial](relbench_example.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/relbench_example.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/relbench_example.ipynb){ .md-button }

-   :material-history: __Recurrent Event Network (RENet) on Temporal Knowledge Graphs__

    ---

    Predicting future dynamic links using recurrent graph convolutional networks.

    - **Backend**: Multi-Backend
    - **Dataset**: `ICEWS18`
    - **Key Layer**: `RENet`

    [:octicons-arrow-right-24: Read Tutorial](renet.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/renet.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/renet.ipynb){ .md-button }

-   :material-backup-restore: __Reversible Graph Neural Networks (RevGNN)__

    ---

    Training deep 100+ layer GNNs with constant memory via reversible layers.

    - **Backend**: Multi-Backend
    - **Dataset**: `ogbn-arxiv / Cora`
    - **Key Layer**: `GroupAddRev`

    [:octicons-arrow-right-24: Read Tutorial](rev_gnn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rev_gnn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rev_gnn.ipynb){ .md-button }

-   :material-tag-multiple-outline: __Relational Graph Attention Network (RGAT) on Entities__

    ---

    Relation-specific attention mechanisms on multi-relational knowledge graphs.

    - **Backend**: Multi-Backend
    - **Dataset**: `Entities (AIFB / MUTAG)`
    - **Key Layer**: `RGATConv`

    [:octicons-arrow-right-24: Read Tutorial](rgat.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rgat.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rgat.ipynb){ .md-button }

-   :material-graph-outline: __Relational Graph Convolutional Network (RGCN) on Entities__

    ---

    Multi-relational message passing with basis and block-diagonal decomposition.

    - **Backend**: Multi-Backend
    - **Dataset**: `Entities (AIFB / MUTAG)`
    - **Key Layer**: `RGCNConv`

    [:octicons-arrow-right-24: Read Tutorial](rgcn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rgcn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rgcn.ipynb){ .md-button }

-   :material-vector-link: __Relational GCN for Link Prediction on FB15k-237__

    ---

    Encoder-decoder knowledge graph completion using RGCN and DistMult score function.

    - **Backend**: Multi-Backend
    - **Dataset**: `FB15k-237`
    - **Key Layer**: `RGCNConv / DistMult`

    [:octicons-arrow-right-24: Read Tutorial](rgcn_link_pred.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rgcn_link_pred.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rgcn_link_pred.ipynb){ .md-button }

-   :material-stamp: __SEAL Subgraph-Based Link Prediction__

    ---

    Enclosing subgraph extraction and node labeling for accurate link prediction.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `DGCNN / SortAggregation`

    [:octicons-arrow-right-24: Read Tutorial](seal_link_pred.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/seal_link_pred.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/seal_link_pred.ipynb){ .md-button }

-   :material-feather: __Simplifying Graph Convolutional Networks (SGConv) on Cora__

    ---

    Linearized graph convolution removing nonlinearities between consecutive layers.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `SGConv`

    [:octicons-arrow-right-24: Read Tutorial](sgc.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/sgc.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/sgc.ipynb){ .md-button }

-   :material-box-shadow: __ShaDow-GNN: Decoupled Subgraph Depth and Scope__

    ---

    Mini-batch GNN training on shallow and deep target-node-centric subgraphs.

    - **Backend**: Multi-Backend
    - **Dataset**: `Flickr`
    - **Key Layer**: `SAGEConv`

    [:octicons-arrow-right-24: Read Tutorial](shadow.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/shadow.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/shadow.ipynb){ .md-button }

-   :material-speedometer: __Scalable Inception Graph Neural Networks (SIGN) on Cora__

    ---

    Precomputing multi-scale graph diffusion operators for fast parallel GNN training.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `SIGN`

    [:octicons-arrow-right-24: Read Tutorial](sign.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/sign.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/sign.ipynb){ .md-button }

-   :material-plus-minus: __Signed Graph Convolutional Network on BitcoinOTC__

    ---

    Modeling positive and negative social interactions inspired by balance theory.

    - **Backend**: Multi-Backend
    - **Dataset**: `BitcoinOTC`
    - **Key Layer**: `SignedGCN`

    [:octicons-arrow-right-24: Read Tutorial](signed_gcn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/signed_gcn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/signed_gcn.ipynb){ .md-button }

-   :material-shield-star-outline: __Self-Supervised Graph Attention Network (SuperGAT) on Cora__

    ---

    Self-supervised edge agreement objective guiding graph attention weights.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `SuperGATConv`

    [:octicons-arrow-right-24: Read Tutorial](super_gat.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/super_gat.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/super_gat.ipynb){ .md-button }

-   :material-lan: __Topology Adaptive Graph Convolutional Networks (TAGCN) on Cora__

    ---

    Polynomial filter graphs learning simultaneous neighborhood representations across hops.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `TAGConv`

    [:octicons-arrow-right-24: Read Tutorial](tagcn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tagcn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/tagcn.ipynb){ .md-button }

-   :material-chart-line: __GNN Experiment Tracking with TensorBoard__

    ---

    Tracking training losses, validation metrics, and embeddings across training epochs.

    - **Backend**: Multi-Backend
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `GCNConv`

    [:octicons-arrow-right-24: Read Tutorial](tensorboard_logging.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tensorboard_logging.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/tensorboard_logging.ipynb){ .md-button }

-   :material-google: __Node Classification on OGBN-Arxiv with ARMAConv__

    ---

    Large-scale node classification on the `ogbn-arxiv` citation benchmark using K3-Node's `ARMAConv` layer, Spektral graph preprocessing, and a custom TensorFlow training loop.

    - **Backend**: TensorFlow
    - **Dataset**: `ogbn-arxiv`
    - **Key Layer**: `ARMAConv`

    [:octicons-arrow-right-24: Read Tutorial](ogb_arxiv_spektral_dataset.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb){ .md-button }

-   :material-timeline-clock-outline: __Temporal Graph Networks (TGN) on Dynamic Continuous-Time Graphs__

    ---

    Continuous-time dynamic graph learning using node memory modules and time encoders.

    - **Backend**: Multi-Backend
    - **Dataset**: `JODIE`
    - **Key Layer**: `TGNMemory`

    [:octicons-arrow-right-24: Read Tutorial](tgn.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tgn.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/tgn.ipynb){ .md-button }

-   :material-fire: __Node Classification on Cora with GatedGraphConv__

    ---

    Node classification on the standard `Planetoid Cora` citation graph using K3-Node's `GatedGraphConv` layer, PyTorch Geometric dataset loading, and PyTorch backend optimization.

    - **Backend**: PyTorch
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `GatedGraphConv`

    [:octicons-arrow-right-24: Read Tutorial](planetoid_PyTorch_Geometric.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb){ .md-button }

-   :material-triangle-outline: __Self-Attention Graph Pooling (SAGPooling) on Triangles__

    ---

    Hierarchical self-attention graph pooling with GNN-computed node importance scores.

    - **Backend**: Multi-Backend
    - **Dataset**: `Triangles / PROTEINS`
    - **Key Layer**: `SAGPooling`

    [:octicons-arrow-right-24: Read Tutorial](triangles_sag_pool.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/triangles_sag_pool.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/triangles_sag_pool.ipynb){ .md-button }

-   :material-swap-horizontal: __Unified Message Passing (UniMP) on OGBN-Arxiv__

    ---

    Combining masked node label propagation with feature-based graph transformers.

    - **Backend**: Multi-Backend
    - **Dataset**: `ogbn-arxiv`
    - **Key Layer**: `TransformerConv`

    [:octicons-arrow-right-24: Read Tutorial](unimp_arxiv.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/unimp_arxiv.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/unimp_arxiv.ipynb){ .md-button }

-   :material-newspaper-variant-outline: __User Preference-Aware Fake News Detection (UPFD)__

    ---

    Hierarchical graph classification modeling news propagation trees and user profiles.

    - **Backend**: Multi-Backend
    - **Dataset**: `UPFD`
    - **Key Layer**: `GCNConv / SAGEConv`

    [:octicons-arrow-right-24: Read Tutorial](upfd.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/upfd.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/upfd.ipynb){ .md-button }

-   :material-graph: __Weisfeiler-Lehman Graph Kernel (WLConv) on MUTAG__

    ---

    Color refinement algorithm for Weisfeiler-Lehman graph isomorphism testing.

    - **Backend**: Multi-Backend
    - **Dataset**: `MUTAG (TUDataset)`
    - **Key Layer**: `WLConv`

    [:octicons-arrow-right-24: Read Tutorial](wl_kernel.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/wl_kernel.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/wl_kernel.ipynb){ .md-button }

</div>

---

## Summary Table

| Backend | Example | Dataset | Key Layer | Colab | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Multi-Backend** | [Attention-based Graph Neural Network (AGNN) on Cora](agnn.md) | `Cora (Planetoid)` | `AGNNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/agnn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/agnn.ipynb) |
| **Multi-Backend** | [Attract-Repel Link Prediction on Cora](ar_link_pred.md) | `Cora (Planetoid)` | `ARLinkPredictor` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/ar_link_pred.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/ar_link_pred.ipynb) |
| **Multi-Backend** | [Adversarially Regularized Variational Graph Autoencoder (ARGVA)](argva_node_clustering.md) | `Cora (Planetoid)` | `ARGVA` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/argva_node_clustering.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/argva_node_clustering.ipynb) |
| **Multi-Backend** | [Auto-Regressive Moving Average Graph Convolution (ARMAConv) on Cora](arma.md) | `Cora (Planetoid)` | `ARMAConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/arma.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/arma.ipynb) |
| **Multi-Backend** | [AttentiveFP Molecular Property Prediction](attentive_fp.md) | `MoleculeNet` | `AttentiveFP` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/attentive_fp.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/attentive_fp.ipynb) |
| **Multi-Backend** | [Graph Autoencoders (GAE & VGAE) on Cora](autoencoder.md) | `Cora (Planetoid)` | `GAE / VGAE` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/autoencoder.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/autoencoder.ipynb) |
| **Multi-Backend** | [Cluster-GCN on PPI Graph Dataset](cluster_gcn_ppi.md) | `PPI` | `SAGEConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/cluster_gcn_ppi.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/cluster_gcn_ppi.ipynb) |
| **Multi-Backend** | [Cluster-GCN on Reddit Graph](cluster_gcn_reddit.md) | `Reddit` | `SAGEConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/cluster_gcn_reddit.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/cluster_gcn_reddit.ipynb) |
| **Multi-Backend** | [Graph Classification with TopKPooling on Colors Dataset](colors_topk_pool.md) | `TUDataset (Colors)` | `TopKPooling` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/colors_topk_pool.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/colors_topk_pool.ipynb) |
| **Multi-Backend** | [Node Classification on Cora Benchmark](cora.md) | `Cora (Planetoid)` | `SplineConv / GCNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/cora.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/cora.ipynb) |
| **Multi-Backend** | [Correct and Smooth (C&S) Post-Processing on Cora](correct_and_smooth.md) | `Cora (Planetoid)` | `CorrectAndSmooth` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/correct_and_smooth.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/correct_and_smooth.ipynb) |
| **Multi-Backend** | [PyG Graph DataPipe & Streaming Loaders](datapipe.md) | `Synthetic Meshes` | `DataLoader` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/datapipe.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/datapipe.ipynb) |
| **Multi-Backend** | [Dynamic Graph CNN (DGCNN) for Point Cloud Classification](dgcnn_classification.md) | `ModelNet / MedShapeNet` | `DynamicEdgeConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/dgcnn_classification.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/dgcnn_classification.ipynb) |
| **Multi-Backend** | [Dynamic Graph CNN for 3D Part Segmentation](dgcnn_segmentation.md) | `ShapeNet` | `DynamicEdgeConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/dgcnn_segmentation.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/dgcnn_segmentation.ipynb) |
| **Multi-Backend** | [Directed Graph Neural Network (DirGNN) on Web Graphs](dir_gnn.md) | `WikipediaNetwork` | `DirGNNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/dir_gnn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/dir_gnn.ipynb) |
| **Multi-Backend** | [Dynamic Neighborhood Aggregation (DNAConv) on Cora](dna.md) | `Cora (Planetoid)` | `DNAConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/dna.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/dna.ipynb) |
| **Multi-Backend** | [Efficient Graph Convolution (EGConv) on Cora](egc.md) | `Cora (Planetoid)` | `EGConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/egc.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/egc.ipynb) |
| **Multi-Backend** | [Equilibrium Aggregation on Graph Benchmarks](equilibrium_median.md) | `Cora` | `EquilibriumAggregation` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/equilibrium_median.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/equilibrium_median.ipynb) |
| **Multi-Backend** | [SplineConv on FAUST 3D Mesh Registration](faust.md) | `FAUST` | `SplineConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/faust.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/faust.ipynb) |
| **Multi-Backend** | [Feature-wise Linear Modulation (FiLMConv) on Cora](film.md) | `Cora (Planetoid)` | `FiLMConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/film.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/film.ipynb) |
| **Multi-Backend** | [Graph Attention Networks (GAT & GATv2) on Cora](gat.md) | `Cora (Planetoid)` | `GATConv / GATv2Conv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/gat.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/gat.ipynb) |
| **Multi-Backend** | [Graph Convolutional Network (GCN) on Cora](gcn.md) | `Cora (Planetoid)` | `GCNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/gcn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/gcn.ipynb) |
| **Multi-Backend** | [Deep GCN with Initial Residual Connections (GCNII) on Cora](gcn2_cora.md) | `Cora (Planetoid)` | `GCN2Conv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/gcn2_cora.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/gcn2_cora.ipynb) |
| **Multi-Backend** | [GCNII on Protein-Protein Interaction (PPI) Dataset](gcn2_ppi.md) | `PPI` | `GCN2Conv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/gcn2_ppi.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/gcn2_ppi.ipynb) |
| **Multi-Backend** | [Adaptive Neighborhood Exploration with GeniePath](geniepath.md) | `PPI / Cora` | `GeniePathConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/geniepath.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/geniepath.ipynb) |
| **Multi-Backend** | [Graph-to-MLP Knowledge Distillation (GLNN)](glnn.md) | `Cora (Planetoid)` | `GCN / MLP` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/glnn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/glnn.ipynb) |
| **Multi-Backend** | [Graph Positional and Structural Embeddings (GPSE)](gpse.md) | `ZINC` | `GPSE` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/gpse.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/gpse.ipynb) |
| **Multi-Backend** | [General Powerful Scalable Graph Transformer (GPS)](graph_gps.md) | `ZINC` | `GPSLayer` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graph_gps.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graph_gps.ipynb) |
| **Multi-Backend** | [Unsupervised GraphSAGE on Citation Network](graph_sage_unsup.md) | `Cora / KarateClub` | `SAGEConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graph_sage_unsup.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graph_sage_unsup.ipynb) |
| **Multi-Backend** | [Unsupervised GraphSAGE on PPI Dataset](graph_sage_unsup_ppi.md) | `PPI` | `SAGEConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graph_sage_unsup_ppi.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graph_sage_unsup_ppi.ipynb) |
| **Multi-Backend** | [GraphSAINT Inductive Subgraph Sampling on Reddit](graph_saint.md) | `Reddit` | `GraphSAINT / SAGEConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graph_saint.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graph_saint.ipynb) |
| **Multi-Backend** | [Graph U-Net with gPool and gUnpool](graph_unet.md) | `Cora (Planetoid)` | `GraphUNet / TopKPooling` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graph_unet.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graph_unet.ipynb) |
| **Multi-Backend** | [GraphLand Benchmark Pipeline](graphland.md) | `GraphLand` | `GCNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/graphland.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/graphland.ipynb) |
| **Multi-Backend** | [Hierarchical Neighborhood Sampling](hierarchical_sampling.md) | `Reddit / Flickr` | `NeighborLoader` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/hierarchical_sampling.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/hierarchical_sampling.ipynb) |
| **Multi-Backend** | [Deep Graph Infomax (Inductive PPI)](infomax_inductive.md) | `PPI` | `DeepGraphInfomax` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/infomax_inductive.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/infomax_inductive.ipynb) |
| **Multi-Backend** | [Deep Graph Infomax (Transductive Cora)](infomax_transductive.md) | `Cora (Planetoid)` | `DeepGraphInfomax` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/infomax_transductive.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/infomax_transductive.ipynb) |
| **Multi-Backend** | [Knowledge Graph Embeddings (TransE / DistMult) on FB15k-237](kge_fb15k_237.md) | `FB15k-237` | `TransE / DistMult` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/kge_fb15k_237.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/kge_fb15k_237.ipynb) |
| **Multi-Backend** | [Label Propagation Algorithm (LPA) on Cora](label_prop.md) | `Cora (Planetoid)` | `LabelPropagation` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/label_prop.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/label_prop.ipynb) |
| **Multi-Backend** | [Custom Aggregation Operators: Second-Min & Order Statistics](lcm_aggr_2nd_min.md) | `Cora` | `Aggregation` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/lcm_aggr_2nd_min.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/lcm_aggr_2nd_min.ipynb) |
| **Multi-Backend** | [LightGCN Recommender System on MovieLens](lightgcn.md) | `MovieLens` | `LightGCN / BPRLoss` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/lightgcn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/lightgcn.ipynb) |
| **Multi-Backend** | [Link Prediction with GCN on Cora](link_pred.md) | `Cora (Planetoid)` | `GCNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/link_pred.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/link_pred.ipynb) |
| **Multi-Backend** | [LINKX on Large Heterophilous Citation Graphs](linkx.md) | `Penn94 / Cora` | `LINKX` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/linkx.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/linkx.ipynb) |
| **Multi-Backend** | [LPFormer: Link Prediction with Transformers](lpformer.md) | `Cora` | `LPFormer` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/lpformer.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/lpformer.ipynb) |
| **Multi-Backend** | [Memory-Based Graph Pooling (MemPooling) on MUTAG](mem_pool.md) | `MUTAG (TUDataset)` | `MemPooling` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mem_pool.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mem_pool.ipynb) |
| **Multi-Backend** | [MixHop: Higher-Order Neighborhood Convolution on Cora](mixhop.md) | `Cora (Planetoid)` | `MixHopConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mixhop.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mixhop.ipynb) |
| **Multi-Backend** | [Superpixel MNIST Classification with Graclus Pooling](mnist_graclus.md) | `MNISTSuperpixels` | `SplineConv / Graclus` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mnist_graclus.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mnist_graclus.ipynb) |
| **Multi-Backend** | [Superpixel MNIST with Continuous Edge Convolutions (NNConv)](mnist_nn_conv.md) | `MNISTSuperpixels` | `NNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mnist_nn_conv.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mnist_nn_conv.ipynb) |
| **Multi-Backend** | [Superpixel MNIST with Voxel Grid Downsampling](mnist_voxel_grid.md) | `MNISTSuperpixels` | `VoxelGrid / SplineConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mnist_voxel_grid.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mnist_voxel_grid.ipynb) |
| **Multi-Backend** | [Graph Isomorphism Network (GIN) on MUTAG](mutag_gin.md) | `MUTAG (TUDataset)` | `GINConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/mutag_gin.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/mutag_gin.ipynb) |
| **Multi-Backend** | [Node2Vec Representation Learning on KarateClub](node2vec.md) | `KarateClub` | `Node2Vec` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/node2vec.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/node2vec.ipynb) |
| **Multi-Backend** | [DeepGCN with Residual Connections on OGBN-Proteins](ogbn_proteins_deepgcn.md) | `ogbn-proteins` | `DeepGCNLayer` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/ogbn_proteins_deepgcn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/ogbn_proteins_deepgcn.ipynb) |
| **Multi-Backend** | [OGBN Graph Benchmark Training Pipeline](ogbn_train.md) | `ogbn-arxiv` | `SAGEConv / GCNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/ogbn_train.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/ogbn_train.ipynb) |
| **Multi-Backend** | [Online Graph Clustering (OGC) with GNNs](ogc.md) | `Reddit` | `ClusterGCNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/ogc.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/ogc.ipynb) |
| **Multi-Backend** | [Propagational MLP (PMLP) on Cora](pmlp.md) | `Cora (Planetoid)` | `PMLP` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/pmlp.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/pmlp.ipynb) |
| **Multi-Backend** | [Principal Neighbourhood Aggregation (PNA) on ZINC](pna.md) | `ZINC` | `PNAConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/pna.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/pna.ipynb) |
| **Multi-Backend** | [Point Transformer for 3D Shape Classification](point_transformer_classification.md) | `ModelNet40` | `PointTransformerConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/point_transformer_classification.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/point_transformer_classification.ipynb) |
| **Multi-Backend** | [Point Transformer for 3D Part Segmentation](point_transformer_segmentation.md) | `ShapeNet` | `PointTransformerConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/point_transformer_segmentation.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/point_transformer_segmentation.ipynb) |
| **Multi-Backend** | [PointNet++ for 3D Point Cloud Classification](pointnet2_classification.md) | `ModelNet40` | `PointNetConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/pointnet2_classification.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/pointnet2_classification.ipynb) |
| **Multi-Backend** | [PointNet++ for 3D Point Cloud Segmentation](pointnet2_segmentation.md) | `ShapeNet` | `PointNetConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/pointnet2_segmentation.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/pointnet2_segmentation.ipynb) |
| **Multi-Backend** | [Inductive Multi-Label Classification with GAT on PPI](ppi.md) | `PPI` | `GATConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/ppi.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/ppi.ipynb) |
| **Multi-Backend** | [Hierarchical Graph Representation with DiffPool on PROTEINS](proteins_diff_pool.md) | `PROTEINS (TUDataset)` | `DenseDiffPool` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/proteins_diff_pool.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/proteins_diff_pool.ipynb) |
| **Multi-Backend** | [Deep Modular Network (DMoN) Pooling on PROTEINS](proteins_dmon_pool.md) | `PROTEINS (TUDataset)` | `DMoNPooling` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/proteins_dmon_pool.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/proteins_dmon_pool.ipynb) |
| **Multi-Backend** | [Graph Multiset Transformer (GMT) Pooling on PROTEINS](proteins_gmt.md) | `PROTEINS (TUDataset)` | `GraphMultisetTransformer` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/proteins_gmt.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/proteins_gmt.ipynb) |
| **Multi-Backend** | [MinCutPooling on PROTEINS Graph Benchmark](proteins_mincut_pool.md) | `PROTEINS (TUDataset)` | `MinCutPooling` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/proteins_mincut_pool.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/proteins_mincut_pool.ipynb) |
| **Multi-Backend** | [TopKPooling on PROTEINS Graph Classification](proteins_topk_pool.md) | `PROTEINS (TUDataset)` | `TopKPooling` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/proteins_topk_pool.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/proteins_topk_pool.ipynb) |
| **Multi-Backend** | [Molecular Property Prediction on QM9 with NNConv](qm9_nn_conv.md) | `QM9` | `NNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/qm9_nn_conv.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/qm9_nn_conv.ipynb) |
| **Multi-Backend** | [Pretrained DimeNet++ on QM9 Molecular Properties](qm9_pretrained_dimenet.md) | `QM9` | `DimeNetPlusPlus` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/qm9_pretrained_dimenet.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/qm9_pretrained_dimenet.ipynb) |
| **Multi-Backend** | [Pretrained SchNet on QM9 Molecular Benchmarks](qm9_pretrained_schnet.md) | `QM9` | `SchNet` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/qm9_pretrained_schnet.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/qm9_pretrained_schnet.ipynb) |
| **Multi-Backend** | [RandLA-Net for Large-Scale Point Cloud Classification](randlanet_classification.md) | `ModelNet40` | `RandLANet` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/randlanet_classification.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/randlanet_classification.ipynb) |
| **Multi-Backend** | [RandLA-Net for Semantic Point Cloud Segmentation](randlanet_segmentation.md) | `S3DIS / ShapeNet` | `RandLANet` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/randlanet_segmentation.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/randlanet_segmentation.ipynb) |
| **Multi-Backend** | [Relational Deep Learning (RDL) Pipeline](rdl.md) | `Relational Databases` | `HeteroConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rdl.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rdl.ipynb) |
| **Multi-Backend** | [RECT: Convex Hull Node Representation Learning](rect.md) | `Cora / WikipediaNetwork` | `RECT_L` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rect.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rect.ipynb) |
| **Multi-Backend** | [Inductive Node Classification on Reddit with SAGEConv](reddit.md) | `Reddit` | `SAGEConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/reddit.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/reddit.ipynb) |
| **Multi-Backend** | [RelBench Relational Benchmark with GNNs](relbench_example.md) | `RelBench` | `HeteroGNN` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/relbench_example.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/relbench_example.ipynb) |
| **Multi-Backend** | [Recurrent Event Network (RENet) on Temporal Knowledge Graphs](renet.md) | `ICEWS18` | `RENet` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/renet.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/renet.ipynb) |
| **Multi-Backend** | [Reversible Graph Neural Networks (RevGNN)](rev_gnn.md) | `ogbn-arxiv / Cora` | `GroupAddRev` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rev_gnn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rev_gnn.ipynb) |
| **Multi-Backend** | [Relational Graph Attention Network (RGAT) on Entities](rgat.md) | `Entities (AIFB / MUTAG)` | `RGATConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rgat.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rgat.ipynb) |
| **Multi-Backend** | [Relational Graph Convolutional Network (RGCN) on Entities](rgcn.md) | `Entities (AIFB / MUTAG)` | `RGCNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rgcn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rgcn.ipynb) |
| **Multi-Backend** | [Relational GCN for Link Prediction on FB15k-237](rgcn_link_pred.md) | `FB15k-237` | `RGCNConv / DistMult` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/rgcn_link_pred.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/rgcn_link_pred.ipynb) |
| **Multi-Backend** | [SEAL Subgraph-Based Link Prediction](seal_link_pred.md) | `Cora (Planetoid)` | `DGCNN / SortAggregation` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/seal_link_pred.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/seal_link_pred.ipynb) |
| **Multi-Backend** | [Simplifying Graph Convolutional Networks (SGConv) on Cora](sgc.md) | `Cora (Planetoid)` | `SGConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/sgc.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/sgc.ipynb) |
| **Multi-Backend** | [ShaDow-GNN: Decoupled Subgraph Depth and Scope](shadow.md) | `Flickr` | `SAGEConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/shadow.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/shadow.ipynb) |
| **Multi-Backend** | [Scalable Inception Graph Neural Networks (SIGN) on Cora](sign.md) | `Cora (Planetoid)` | `SIGN` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/sign.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/sign.ipynb) |
| **Multi-Backend** | [Signed Graph Convolutional Network on BitcoinOTC](signed_gcn.md) | `BitcoinOTC` | `SignedGCN` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/signed_gcn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/signed_gcn.ipynb) |
| **Multi-Backend** | [Self-Supervised Graph Attention Network (SuperGAT) on Cora](super_gat.md) | `Cora (Planetoid)` | `SuperGATConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/super_gat.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/super_gat.ipynb) |
| **Multi-Backend** | [Topology Adaptive Graph Convolutional Networks (TAGCN) on Cora](tagcn.md) | `Cora (Planetoid)` | `TAGConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tagcn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/tagcn.ipynb) |
| **Multi-Backend** | [GNN Experiment Tracking with TensorBoard](tensorboard_logging.md) | `Cora (Planetoid)` | `GCNConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tensorboard_logging.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/tensorboard_logging.ipynb) |
| **TensorFlow** | [Node Classification on OGBN-Arxiv with ARMAConv](ogb_arxiv_spektral_dataset.md) | `ogbn-arxiv` | `ARMAConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb) |
| **Multi-Backend** | [Temporal Graph Networks (TGN) on Dynamic Continuous-Time Graphs](tgn.md) | `JODIE` | `TGNMemory` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tgn.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/tgn.ipynb) |
| **PyTorch** | [Node Classification on Cora with GatedGraphConv](planetoid_PyTorch_Geometric.md) | `Cora (Planetoid)` | `GatedGraphConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb) |
| **Multi-Backend** | [Self-Attention Graph Pooling (SAGPooling) on Triangles](triangles_sag_pool.md) | `Triangles / PROTEINS` | `SAGPooling` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/triangles_sag_pool.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/triangles_sag_pool.ipynb) |
| **Multi-Backend** | [Unified Message Passing (UniMP) on OGBN-Arxiv](unimp_arxiv.md) | `ogbn-arxiv` | `TransformerConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/unimp_arxiv.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/unimp_arxiv.ipynb) |
| **Multi-Backend** | [User Preference-Aware Fake News Detection (UPFD)](upfd.md) | `UPFD` | `GCNConv / SAGEConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/upfd.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/upfd.ipynb) |
| **Multi-Backend** | [Weisfeiler-Lehman Graph Kernel (WLConv) on MUTAG](wl_kernel.md) | `MUTAG (TUDataset)` | `WLConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/wl_kernel.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/wl_kernel.ipynb) |
