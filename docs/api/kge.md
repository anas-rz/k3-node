# Knowledge Graph Embedding (KGE)

K3-node provides a multi-backend implementation of Knowledge Graph Embedding (KGE) models and negative sampling triplet iterators under `k3_node.layers.kge`.

KGE models learn low-dimensional vector representations for entities $\mathcal{E}$ and relations $\mathcal{R}$ in multi-relational knowledge graphs, scoring triplets $(h, r, t)$ according to score functions $f(h, r, t)$.

---

## Models

### Base Class: KGEModel
`KGEModel` defines the base architecture for knowledge graph embeddings, handling entity/relation lookup, margin ranking loss, and negative sampling scoring.

```python
from k3_node.layers.kge import KGEModel
```

::: k3_node.layers.kge.KGEModel

---

### TransE
Translational distance model ($h + r \approx t$) mapping entities and relations to a real vector space:
$$d(h + r, t) = -\| \mathbf{e}_h + \mathbf{e}_r - \mathbf{e}_t \|_p$$

```python
from k3_node.layers.kge import TransE

model = TransE(num_nodes=1000, num_relations=50, hidden_channels=64, margin=1.0, p_norm=1.0)
```

::: k3_node.layers.kge.TransE

---

### RotatE
Knowledge graph embedding by relational rotation in complex space ($\mathbf{e}_t = \mathbf{e}_h \circ \mathbf{r}$ where $|\mathbf{r}_i| = 1$):
$$d(h \circ r, t) = -\| \mathbf{e}_h \circ \mathbf{e}_r - \mathbf{e}_t \|$$

```python
from k3_node.layers.kge import RotatE

model = RotatE(num_nodes=1000, num_relations=50, hidden_channels=64, margin=6.0)
```

::: k3_node.layers.kge.RotatE

---

### DistMult
Bilinear diagonal model capturing symmetric relation interactions:
$$f(h, r, t) = \langle \mathbf{e}_h, \mathbf{e}_r, \mathbf{e}_t \rangle = \sum_{i} (\mathbf{e}_h)_i (\mathbf{e}_r)_i (\mathbf{e}_t)_i$$

```python
from k3_node.layers.kge import DistMult

model = DistMult(num_nodes=1000, num_relations=50, hidden_channels=64, margin=1.0)
```

::: k3_node.layers.kge.DistMult

---

### ComplEx
Complex embeddings for simple link prediction, handling asymmetric relations via Hermitian dot product:
$$f(h, r, t) = \text{Re}(\langle \mathbf{e}_h, \mathbf{e}_r, \bar{\mathbf{e}}_t \rangle)$$

```python
from k3_node.layers.kge import ComplEx

model = ComplEx(num_nodes=1000, num_relations=50, hidden_channels=64, margin=1.0)
```

::: k3_node.layers.kge.ComplEx

---

## Data Loading & Triplet Batching

### KGTripletLoader
Framework-agnostic batch iterator for knowledge graph triplets $(h, r, t)$ with corrupt head / tail negative sampling:

```python
from k3_node.layers.kge import KGTripletLoader

loader = KGTripletLoader(
    head_index=head_indices,
    rel_type=rel_types,
    tail_index=tail_indices,
    batch_size=256,
    shuffle=True
)

for head, rel, tail in loader:
    loss = model.loss(head, rel, tail)
```

::: k3_node.layers.kge.KGTripletLoader

