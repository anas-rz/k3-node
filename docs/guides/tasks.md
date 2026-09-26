# High-Level Task APIs Guide

K3-Node provides intuitive, scikit-learn-style high-level estimators designed to get you from raw graph data to trained models, predictions, and evaluation in 3 to 5 lines of code.

Whether you run on **TensorFlow**, **PyTorch**, or **JAX**, the Task API abstracts away training boilerplate, tensor dimension alignment, data batching, and loss configurations while giving you full access to all backbones across `k3_node.models`.

---

## 1. Node Classification (`NodeClassifier`)

Classifies nodes in a transductive or inductive graph. Automatically infers `in_channels` and `num_classes` from the dataset, and supports backbones like `"gcn"`, `"gat"`, `"sage"`, `"gin"`, `"pna"`, `"linkx"`, and custom Keras models.

```python
import os
os.environ["KERAS_BACKEND"] = "torch"  # or 'tensorflow', 'jax'

import k3_node
from k3_node.datasets import Planetoid
from k3_node.tasks import NodeClassifier

# 1. Load benchmark dataset
dataset = Planetoid(root="./data/Planetoid", name="Cora")
data = dataset[0]

# 2. Initialize classifier with your chosen backbone
clf = NodeClassifier(backbone="gcn", hidden_channels=64, num_layers=2, dropout=0.5)

# 3. Train on train_mask
clf.fit(data, epochs=20, lr=0.01)

# 4. Predict and evaluate on test_mask
preds = clf.predict(data, mask="test_mask")
metrics = clf.evaluate(data, mask="test_mask")
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
```

### Supported Backbones
- `"gcn"`: Standard graph convolutional network.
- `"gat"`: Graph attention network.
- `"sage"` / `"graphsage"`: Inductive neighborhood sampling and aggregation.
- `"gin"`: Graph isomorphism network with maximal expressive power.
- `"linkx"`: Large-scale sparse graph model.
- Or pass any custom `keras.Model`.

---

## 2. Node Regression (`NodeRegressor`)

Predicts continuous node properties (e.g. node centrality, physical attributes, traffic flow).

```python
from k3_node.tasks import NodeRegressor

reg = NodeRegressor(backbone="gat", hidden_channels=32, num_layers=2)
reg.fit(data, epochs=20, lr=0.01)

predictions = reg.predict(data, mask="test_mask")
eval_results = reg.evaluate(data, mask="test_mask")
print(f"Test MAE: {eval_results['mae']:.4f}, MSE: {eval_results['mse']:.4f}")
```

---

## 3. Graph Classification (`GraphClassifier`)

Classifies whole graphs (e.g., molecule toxicity, protein function, social graphs) by combining a message-passing backbone with global readout pooling (`"mean"`, `"add"`, `"max"`) and a classification head.

```python
from k3_node.datasets import TUDataset
from k3_node.tasks import GraphClassifier

# 1. Load graph dataset
dataset = TUDataset(root="./data/MUTAG", name="MUTAG")
train_data, test_data = dataset[:150], dataset[150:]

# 2. Initialize graph classifier
model = GraphClassifier(
    backbone="gin",
    hidden_channels=64,
    num_layers=3,
    pooling="mean",
    dropout=0.5,
)

# 3. Train across batches
model.fit(train_data, epochs=20, batch_size=32, lr=0.01)

# 4. Evaluate on test set
eval_res = model.evaluate(test_data, batch_size=32)
print(f"Test Accuracy: {eval_res['accuracy']:.4f}")
```

---

## 4. Graph & Molecular Regression (`GraphRegressor`)

Predicts continuous graph-level targets such as quantum chemical molecular energies (HOMO/LUMO gaps, dipole moments) or crystal structure properties. Supports both general GNNs (`"sage"`, `"gin"`, `"pna"`) and domain-specific geometric backbones (`"schnet"`, `"dimenet"`, `"dimenet++"`, `"attentive_fp"`).

```python
from k3_node.tasks import GraphRegressor

# Molecular property prediction
reg = GraphRegressor(
    backbone="schnet",
    hidden_channels=128,
    num_layers=4,
    loss="mae",
)

# Train on molecular dataset (e.g., QM9 or custom molecules)
reg.fit(train_molecules, epochs=30, batch_size=64, lr=1e-3)
test_metrics = reg.evaluate(test_molecules, batch_size=64)
print(f"Test MAE: {test_metrics['mae']:.4f}")
```

---

## 5. Link Prediction (`LinkPredictor`)

Predicts edge existence or relationship probabilities between pairs of nodes. Automatically performs dynamic negative sampling during training and computes ROC-AUC, Average Precision (AP), and binary accuracy.

```python
from k3_node.tasks import LinkPredictor

# 1. Initialize link predictor with dot-product or MLP decoder
lp = LinkPredictor(
    backbone="gcn",
    hidden_channels=64,
    out_channels=32,
    decoder="inner_product",  # or 'cosine', 'mlp'
)

# 2. Train on graph connectivity with automatic negative edge sampling
lp.fit(data, epochs=20, lr=0.01, neg_ratio=1.0)

# 3. Evaluate AUC & AP
results = lp.evaluate(data)
print(f"Accuracy: {results['accuracy']:.4f}")
if "auc" in results:
    print(f"ROC-AUC: {results['auc']:.4f}, AP: {results['ap']:.4f}")

# 4. Predict edge probabilities for query pairs
edge_pairs = [[0, 1, 2], [1, 2, 3]]
probs = lp.predict_proba(data, edge_label_index=edge_pairs)
print(f"Edge probabilities: {probs}")
```

---

## 6. Applications API Integration

For domain-specific tasks, K3-Node groups specialized foundation models and pipelines under `k3_node.applications`:

- `k3_node.applications.chemistry`: Molecular representations (`Uni-Mol`, `AttentiveFP`, `SchNet`, `DimeNetPlusPlus`).
- `k3_node.applications.materials`: Crystal and inorganic material modeling (`MEGNet`, `M3GNet`, `CHGNet`).
- `k3_node.applications.bio`: Macromolecular and protein interaction models (`ESM`, `AlphaFold`-style graph encoders).

These backbones plug directly into `GraphClassifier`, `GraphRegressor`, and `NodeClassifier`!
