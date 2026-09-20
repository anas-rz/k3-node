# Interactive Examples & Complete Code

This page automatically indexes all interactive notebooks from the [`examples/`](https://github.com/anas-rz/k3-node/tree/main/examples) directory.
You can run any notebook directly in **Google Colab**, inspect the notebook on GitHub, or copy the complete, self-contained Python code below.

---

## Summary of Examples

| Backend | Task / Dataset | Layer / Model | Notebook | Colab |
| :--- | :--- | :--- | :--- | :--- |
| **TensorFlow** | Node Classification on OGBN-Arxiv using ARMAConv | `ARMAConv` | [`ogb_arxiv_spektral_dataset.ipynb`](https://github.com/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb) |
| **PyTorch** | Node Classification on Cora using GatedGraphConv | `GatedGraphConv` | [`planetoid_PyTorch_Geometric.ipynb`](https://github.com/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb) |

---

## Node Classification on OGBN-Arxiv using ARMAConv (TensorFlow)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb) &nbsp; [View on GitHub](https://github.com/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb)

**File Location**: [`examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb`](https://github.com/anas-rz/k3-node/blob/main/examples/tensorflow/ogb_arxiv_spektral_dataset.ipynb)

### Complete Runnable Code

```python
import os, sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
sys.path.append('/content/k3-node')

import numpy as np
from ogb.nodeproppred import NodePropPredDataset
from keras.layers import BatchNormalization, Dropout, Input
from keras.losses import SparseCategoricalCrossentropy
from keras.metrics import SparseCategoricalAccuracy
from keras.models import Model
from keras.optimizers import Adam

from spektral.datasets.ogb import OGB
from spektral.transforms import AdjToSpTensor, GCNFilter

from pprint import pprint

from k3_node.layers import ARMAConv

# Load data
dataset_name = "ogbn-arxiv"
ogb_dataset = NodePropPredDataset(dataset_name)
dataset = OGB(ogb_dataset, transforms=[GCNFilter(), AdjToSpTensor()])
graph = dataset[0]
x, adj, y = graph.x, graph.a, graph.y

# Parameters
channels = 256  # Number of channels for GCN layers
dropout = 0.5  # Dropout rate for the features
learning_rate = 1e-2  # Learning rate
epochs = 200  # Number of training epochs
N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = ogb_dataset.num_classes  # OGB labels are sparse indices

# Data splits
idx = ogb_dataset.get_idx_split()
idx_tr, idx_va, idx_te = idx["train"], idx["valid"], idx["test"]
mask_tr = np.zeros(N, dtype=bool)
mask_va = np.zeros(N, dtype=bool)
mask_te = np.zeros(N, dtype=bool)
mask_tr[idx_tr] = True
mask_va[idx_va] = True
mask_te[idx_te] = True
masks = [mask_tr, mask_va, mask_te]

# Model definition
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True)
x_1 = ARMAConv(channels, activation="relu")([x_in, a_in])
x_1 = BatchNormalization()(x_1)
x_1 = Dropout(dropout)(x_1)
x_2 = ARMAConv(channels, activation="relu")([x_1, a_in])
x_2 = BatchNormalization()(x_2)
x_2 = Dropout(dropout)(x_2)
x_3 = ARMAConv(n_out, activation="softmax")([x_2, a_in])

# Build model
model = Model(inputs=[x_in, a_in], outputs=x_3)
optimizer = Adam(learning_rate=learning_rate)
loss_fn = SparseCategoricalCrossentropy()
acc_metric = SparseCategoricalAccuracy()
model.summary()

import tensorflow as tf
# Training function
@tf.function
def train(inputs, target, mask):
    acc_metric.reset_state()
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target[mask], predictions[mask]) + sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc_metric.update_state(target[mask], predictions[mask])
    return loss, acc_metric.result()

@tf.function
def evaluate(inputs, target, mask):
    acc_metric.reset_state()
    predictions = model(inputs, training=True)
    loss = loss_fn(target[mask], predictions[mask]) + sum(model.losses)
    acc_metric.update_state(target[mask], predictions[mask])
    return loss, acc_metric.result()

# Train model
for i in range(1, 1 + epochs):
    tr_loss, tr_acc = train([x, adj], y, mask_tr)
    eval_loss, eval_acc = evaluate([x, adj], y, mask_va) # TODO Add more metrics
    pprint(f"EPOCH {i}: Training Loss {tr_loss.numpy()} - Training Accuracy {tr_acc}, Validation Loss: {eval_loss} - Validation Accuracy {eval_acc}")
test_loss, test_acc = evaluate([x, adj], y, mask_te)
pprint(f"Test Loss: {test_loss} Test Accuracy: {test_acc}")
```

---

## Node Classification on Cora using GatedGraphConv (PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb) &nbsp; [View on GitHub](https://github.com/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb)

**File Location**: [`examples/torch/planetoid_PyTorch_Geometric.ipynb`](https://github.com/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb)

### Complete Runnable Code

```python
import os, sys
sys.path.append('./k3-node')
os.environ['KERAS_BACKEND'] = 'torch'

import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid

import keras
from keras import layers, Model, ops


from k3_node.layers import GatedGraphConv
from k3_node.utils import edge_index_to_adjacency_matrix

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath('.')), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]

class Net(Model):
    def __init__(self):
        super().__init__()
        self.dropout1 = layers.Dropout(0.3)
        self.dropout2 = layers.Dropout(0.3)
        self.lin1 = layers.Dense(16)
        self.prop1 = GatedGraphConv(128, 3)
        self.prop2 = GatedGraphConv(128, 2)
        self.lin2 = layers.Dense(dataset.num_classes)

    def call(self, data=None):
        x = data.x
        adj = edge_index_to_adjacency_matrix(data.edge_index)
        x = self.dropout1(x)
        x = ops.relu(self.lin1(x))

        x = self.prop1((x, adj))
        x = self.prop2((x, adj))
        x = self.dropout2(x)
        x = self.lin2(x)
        return ops.log_softmax(x, axis=1)

@torch.no_grad()
def test(model):
    model.eval()
    out, accs = model(data=data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

model = Net()
optimizer = keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
best_val_acc = 0
for epoch in range(1, 51):
    # Forward pass
    out = model(data=data)
    loss = loss_fn(data.y[data.train_mask], out[data.train_mask])

    # Backward pass
    model.zero_grad()
    trainable_weights = [v for v in model.trainable_weights]

    # Call torch.Tensor.backward() on the loss to compute gradients
    # for the weights.
    loss.backward()
    gradients = [v.value.grad for v in trainable_weights]

    # Update weights
    with torch.no_grad():
        optimizer.apply(gradients, trainable_weights)

    train_acc, val_acc, tmp_test_acc = test(model)

    print(
        f"Training loss at epoch {epoch}: {loss.detach().numpy():.4f}"
    )
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, '
          f'Val: {best_val_acc:.4f}, Test: {test_acc:.4f}')
```

---
