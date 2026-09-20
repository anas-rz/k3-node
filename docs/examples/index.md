# Code Examples

Welcome to the **K3-Node Code Examples**. This page indexes end-to-end runnable tutorials demonstrating how to build, train, and evaluate Graph Neural Networks with K3-Node across multiple frameworks and backends.

---

## Available Examples

<div class="grid cards" markdown>

-   :material-google: __Node Classification on OGBN-Arxiv with ARMAConv__

    ---

    Large-scale node classification on the `ogbn-arxiv` citation benchmark using K3-Node's `ARMAConv` layer, Spektral graph preprocessing, and a custom TensorFlow training loop.

    - **Backend**: TensorFlow
    - **Dataset**: `ogbn-arxiv`
    - **Key Layer**: `ARMAConv`

    [:octicons-arrow-right-24: Read Tutorial](ogb_arxiv_spektral_dataset.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb){ .md-button }

-   :material-fire: __Node Classification on Cora with GatedGraphConv__

    ---

    Node classification on the standard `Planetoid Cora` citation graph using K3-Node's `GatedGraphConv` layer, PyTorch Geometric dataset loading, and PyTorch backend optimization.

    - **Backend**: PyTorch
    - **Dataset**: `Cora (Planetoid)`
    - **Key Layer**: `GatedGraphConv`

    [:octicons-arrow-right-24: Read Tutorial](planetoid_PyTorch_Geometric.md){ .md-button .md-button--primary } &nbsp; [:simple-googlecolab: View in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb){ .md-button } &nbsp; [:octicons-mark-github-16: GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb){ .md-button }

</div>

---

## Summary Table

| Backend | Example | Dataset | Key Layer | Colab | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TensorFlow** | [Node Classification on OGBN-Arxiv with ARMAConv](ogb_arxiv_spektral_dataset.md) | `ogbn-arxiv` | `ARMAConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb) |
| **PyTorch** | [Node Classification on Cora with GatedGraphConv](planetoid_PyTorch_Geometric.md) | `Cora (Planetoid)` | `GatedGraphConv` | [Open in Colab](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb) | [GitHub source](https://github.com/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb) |
