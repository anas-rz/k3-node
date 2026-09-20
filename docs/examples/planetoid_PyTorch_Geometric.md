# Node Classification on Cora with GatedGraphConv

**Author:** K3-Node Team<br>
**Backend:** PyTorch<br>
**Dataset:** `Cora (Planetoid)`<br>
**Description:** Node classification on the standard `Planetoid Cora` citation graph using K3-Node's `GatedGraphConv` layer, PyTorch Geometric dataset loading, and PyTorch backend optimization.

[:simple-googlecolab: **View in Colab**](https://colab.research.google.com/github/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb){ .md-button .md-button--primary } &nbsp; [:octicons-mark-github-16: **GitHub source**](https://github.com/anas-rz/k3-node/blob/main/examples/torch/planetoid_PyTorch_Geometric.ipynb){ .md-button }

---

## Setup & Installation

```bash
pip install torch_geometric
pip install git+http://github.com/anas-rz/k3-node/
```

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
```

```python
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
```

```python
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
```

```python
@torch.no_grad()
def test(model):
    model.eval()
    out, accs = model(data=data), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = out[mask].argmax(1)
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
```

```python
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

??? example "View Output"
    ```text
    Training loss at epoch 1: 1.9491
    Epoch: 001, Train: 0.1571, Val: 0.1620, Test: 0.1450
    Training loss at epoch 2: 1.8746
    Epoch: 002, Train: 0.2714, Val: 0.1840, Test: 0.1890
    Training loss at epoch 3: 1.8219
    Epoch: 003, Train: 0.2357, Val: 0.1840, Test: 0.1890
    Training loss at epoch 4: 1.7723
    Epoch: 004, Train: 0.3357, Val: 0.2980, Test: 0.3100
    Training loss at epoch 5: 1.7175
    Epoch: 005, Train: 0.3500, Val: 0.3360, Test: 0.3440
    Training loss at epoch 6: 1.6714
    Epoch: 006, Train: 0.3500, Val: 0.3360, Test: 0.3440
    Training loss at epoch 7: 1.6145
    Epoch: 007, Train: 0.3643, Val: 0.3360, Test: 0.3440
    Training loss at epoch 8: 1.5644
    Epoch: 008, Train: 0.3857, Val: 0.3660, Test: 0.3690
    Training loss at epoch 9: 1.5077
    Epoch: 009, Train: 0.4214, Val: 0.4300, Test: 0.4280
    Training loss at epoch 10: 1.4584
    Epoch: 010, Train: 0.4071, Val: 0.4380, Test: 0.4300
    Training loss at epoch 11: 1.4119
    Epoch: 011, Train: 0.4357, Val: 0.4520, Test: 0.4390
    Training loss at epoch 12: 1.3715
    Epoch: 012, Train: 0.4571, Val: 0.4520, Test: 0.4390
    Training loss at epoch 13: 1.3358
    Epoch: 013, Train: 0.4571, Val: 0.4520, Test: 0.4390
    Training loss at epoch 14: 1.3045
    Epoch: 014, Train: 0.4500, Val: 0.4540, Test: 0.4310
    Training loss at epoch 15: 1.2714
    Epoch: 015, Train: 0.4857, Val: 0.4620, Test: 0.4370
    Training loss at epoch 16: 1.2346
    Epoch: 016, Train: 0.4714, Val: 0.4620, Test: 0.4370
    Training loss at epoch 17: 1.2026
    Epoch: 017, Train: 0.4643, Val: 0.4700, Test: 0.4290
    Training loss at epoch 18: 1.2173
    Epoch: 018, Train: 0.5286, Val: 0.4980, Test: 0.4760
    Training loss at epoch 19: 1.1394
    Epoch: 019, Train: 0.5500, Val: 0.5060, Test: 0.4860
    Training loss at epoch 20: 1.0966
    Epoch: 020, Train: 0.6000, Val: 0.5060, Test: 0.4860
    Training loss at epoch 21: 1.0769
    Epoch: 021, Train: 0.6286, Val: 0.5240, Test: 0.5130
    Training loss at epoch 22: 1.0197
    Epoch: 022, Train: 0.6143, Val: 0.5240, Test: 0.5130
    Training loss at epoch 23: 1.0088
    Epoch: 023, Train: 0.6429, Val: 0.5300, Test: 0.5390
    Training loss at epoch 24: 0.9583
    Epoch: 024, Train: 0.6571, Val: 0.5560, Test: 0.5380
    Training loss at epoch 25: 0.9295
    Epoch: 025, Train: 0.6714, Val: 0.5620, Test: 0.5490
    Training loss at epoch 26: 0.8863
    Epoch: 026, Train: 0.6857, Val: 0.5640, Test: 0.5680
    Training loss at epoch 27: 0.8549
    Epoch: 027, Train: 0.7143, Val: 0.5640, Test: 0.5680
    Training loss at epoch 28: 0.8141
    Epoch: 028, Train: 0.7429, Val: 0.5640, Test: 0.5680
    Training loss at epoch 29: 0.7707
    Epoch: 029, Train: 0.7500, Val: 0.5700, Test: 0.5660
    Training loss at epoch 30: 0.7450
    Epoch: 030, Train: 0.7786, Val: 0.5700, Test: 0.5660
    Training loss at epoch 31: 0.6935
    Epoch: 031, Train: 0.7929, Val: 0.5700, Test: 0.5660
    Training loss at epoch 32: 0.6676
    Epoch: 032, Train: 0.7857, Val: 0.5720, Test: 0.5740
    Training loss at epoch 33: 0.6315
    Epoch: 033, Train: 0.7857, Val: 0.5720, Test: 0.5740
    Training loss at epoch 34: 0.6029
    Epoch: 034, Train: 0.8143, Val: 0.5720, Test: 0.5740
    Training loss at epoch 35: 0.5699
    Epoch: 035, Train: 0.8214, Val: 0.5820, Test: 0.5690
    Training loss at epoch 36: 0.5380
    Epoch: 036, Train: 0.8357, Val: 0.5860, Test: 0.5770
    Training loss at epoch 37: 0.5094
    Epoch: 037, Train: 0.8500, Val: 0.5980, Test: 0.5850
    Training loss at epoch 38: 0.4796
    Epoch: 038, Train: 0.8429, Val: 0.5980, Test: 0.5850
    Training loss at epoch 39: 0.4531
    Epoch: 039, Train: 0.8786, Val: 0.5980, Test: 0.5850
    Training loss at epoch 40: 0.4267
    Epoch: 040, Train: 0.8643, Val: 0.5980, Test: 0.5850
    Training loss at epoch 41: 0.4058
    Epoch: 041, Train: 0.8857, Val: 0.5980, Test: 0.5850
    Training loss at epoch 42: 0.3740
    Epoch: 042, Train: 0.8786, Val: 0.5980, Test: 0.5850
    Training loss at epoch 43: 0.3621
    Epoch: 043, Train: 0.8786, Val: 0.5980, Test: 0.5850
    Training loss at epoch 44: 0.3614
    Epoch: 044, Train: 0.8929, Val: 0.5980, Test: 0.5850
    Training loss at epoch 45: 0.3170
    Epoch: 045, Train: 0.8786, Val: 0.5980, Test: 0.5850
    Training loss at epoch 46: 0.3351
    Epoch: 046, Train: 0.8500, Val: 0.5980, Test: 0.5850
    Training loss at epoch 47: 0.4243
    Epoch: 047, Train: 0.8214, Val: 0.5980, Test: 0.5850
    Training loss at epoch 48: 0.4371
    Epoch: 048, Train: 0.8643, Val: 0.6180, Test: 0.6350
    Training loss at epoch 49: 0.3429
    Epoch: 049, Train: 0.8857, Val: 0.6180, Test: 0.6350
    Training loss at epoch 50: 0.3071
    Epoch: 050, Train: 0.8714, Val: 0.6180, Test: 0.6350
    ```
