# Attention Layers

The `k3_node.layers.attention` module provides standalone attention building blocks used by K3-Node's scalable graph transformers (`Polynormer`, `SGFormer`, `GPSE`) and can also be composed into custom architectures. Unlike `k3_node.layers.conv` attention layers (`GATConv`, `TransformerConv`, ...), these operate on dense `[batch, nodes, channels]` tensors rather than sparse `edge_index` message passing.

---

## Linear / Kernelized Attention

### PerformerAttention
FAVOR+ kernelized linear attention (from the [Performer](https://arxiv.org/abs/2009.14794) paper), used by `Polynormer` for its global attention branch. Approximates full softmax attention in linear time/memory by projecting queries/keys through random Fourier features.

::: k3_node.layers.attention.PerformerAttention

### PerformerProjection
The random-feature projection used internally by `PerformerAttention` to approximate the softmax kernel; useful standalone when building a custom linear-attention layer.

::: k3_node.layers.attention.PerformerProjection

**Usage:**
```python
from keras import ops
from k3_node import layers as k3_layers

attn = k3_layers.PerformerAttention(channels=64, heads=4)
mask = ops.ones((1, num_nodes))  # [batch, nodes]
out = attn(x, mask)  # x: [batch, nodes, 64]
```

---

## Polynomial & Scalable Graph Attention

### PolynormerAttention
Local-to-global polynomial-expressive attention from [Polynormer](https://arxiv.org/abs/2403.01232), combining a local propagation term with a global attention term whose polynomial expansion is computed via `PerformerAttention`-style kernelization.

::: k3_node.layers.attention.PolynormerAttention

### SGFormerAttention
All-pair, single-layer linear attention from [SGFormer](https://arxiv.org/abs/2306.10759), designed to replace deep attention stacks with one global mixing layer for scalable graph transformers.

::: k3_node.layers.attention.SGFormerAttention

**Usage:**
```python
attn = k3_layers.SGFormerAttention(channels=64, heads=4)
out = attn(x)  # x: [batch, nodes, 64]
```

---

## Q-Former (Query Transformer)

### QFormer
A BERT-style querying transformer (as used in BLIP-2 / `GPSE`) that distills a variable-length input sequence into a fixed set of learned query tokens via cross-attention, useful for pooling variable-size node sets into a fixed-size graph representation.

::: k3_node.layers.attention.QFormer

### QFormerEncoderLayer
A single self-attention + feed-forward encoder block used inside `QFormer`.

::: k3_node.layers.attention.QFormerEncoderLayer

**Usage:**
```python
qformer = k3_layers.QFormer(input_dim=64, hidden_dim=64, output_dim=32, num_heads=4, num_layers=2)
out = qformer(x)  # x: [batch, nodes, 64] -> [batch, nodes, 32]
```
