# Quickstart Guide

This guide introduces the core concepts of **K3 Node**: creating graph data, defining neural network layers, training models using standard Keras 3 APIs, and loading pre-trained weights.

---

## 1. Graph Representation in K3 Node

Graphs in K3 Node are represented using node feature matrices $X$ and coordinate list (COO) edge indices $A$:

- **Node Features (`x`)**: A 2D tensor of shape `(num_nodes, in_channels)`.
- **Edge Index (`edge_index`)**: A 2D integer tensor of shape `(2, num_edges)`, where `edge_index[0]` represents source nodes and `edge_index[1]` represents target nodes.
- **Edge Attributes (`edge_attr`)** *(optional)*: A 2D tensor of shape `(num_edges, edge_channels)`.
- **Batch Vector (`batch`)** *(optional)*: A 1D integer tensor assigning each node to its graph in a disconnected mini-batch.

### Using the `Data` Container

```python
from k3_node.data import Data
import numpy as np
from keras import ops

x = ops.convert_to_tensor(np.random.randn(4, 16).astype(np.float32))
edge_index = ops.convert_to_tensor(np.array([[0, 1, 2, 3], [1, 2, 3, 0]]), dtype="int64")

data = Data(x=x, edge_index=edge_index)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
```

---

## 2. Defining a GNN Layer

K3 Node provides the [`MessagePassing`](api/conv.md) base class to implement custom spatial graph convolutions:

```python
from k3_node.layers.conv import MessagePassing
from keras import layers, ops

class CustomGraphConv(MessagePassing):
    def __init__(self, out_channels, **kwargs):
        super().__init__(aggr="add", **kwargs)
        self.lin = layers.Dense(out_channels)

    def call(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.lin(aggr_out)
```

---

## 3. Building and Training a Model

Because K3 Node models are native `keras.Model` instances, you can use Keras 3 `compile` and `fit`, or write custom training loops:

```python
import keras
from keras import layers, ops
from k3_node.layers.conv import GATConv
from k3_node.layers.pool import global_mean_pool

class GATClassifier(keras.Model):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim, heads=1)
        self.fc = layers.Dense(out_dim)

    def call(self, inputs):
        x, edge_index, batch = inputs
        x = ops.elu(self.conv1(x, edge_index))
        x = ops.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(x)

model = GATClassifier(hidden_dim=32, out_dim=2)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
)
```

---

## 4. Multi-Graph Batching

To batch multiple graphs into a single disjoint graph for parallel processing:

```python
from k3_node.data import Data, Batch
import numpy as np
from keras import ops

g1 = Data(
    x=ops.convert_to_tensor(np.ones((3, 4), dtype=np.float32)),
    edge_index=ops.convert_to_tensor([[0, 1], [1, 2]], dtype="int64"),
)
g2 = Data(
    x=ops.convert_to_tensor(np.ones((2, 4), dtype=np.float32) * 2),
    edge_index=ops.convert_to_tensor([[0], [1]], dtype="int64"),
)

batch = Batch.from_data_list([g1, g2])
print("Batch x shape:", ops.shape(batch.x))          # (5, 4)
print("Batch edge_index shape:", ops.shape(batch.edge_index))  # (2, 3)
print("Batch vector:", ops.convert_to_numpy(batch.batch))      # [0, 0, 0, 1, 1]
```

