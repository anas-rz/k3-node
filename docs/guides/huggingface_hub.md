# Hugging Face Hub Integration

**K3-Node** provides first-class, native integration with the [Hugging Face Hub](https://huggingface.co/), allowing you to share, discover, and download Graph Neural Network models and graph datasets with a single line of code.

Because K3-Node is built on **Keras 3**, every model pushed to the Hub is truly **multi-backend**: you can upload a model trained with PyTorch and load it instantly for inference with JAX or TensorFlow, and vice versa!

---

## Key Features

- **One-line Sharing & Loading**: `model.push_to_hub("username/cora-gcn")` and `NodeClassifier.from_pretrained("username/cora-gcn")`.
- **Automatic Model Cards**: Automatically generates rich `README.md` model cards with metadata, `graph-ml` pipeline tags, hyperparameters, metrics, and copy-paste usage snippets.
- **Graph Dataset Hub**: Push and load custom graph datasets (`Data` or `List[Data]`) directly to/from Hugging Face dataset repositories.
- **Universal Multi-Backend Compatibility**: Saved weights and configs work across PyTorch, JAX, and TensorFlow backends.

---

## 1. Saving & Loading Models Locally

Before uploading to the Hub, you can save and load model checkpoints locally using the exact same standard Hugging Face format:

```python
import os
os.environ["KERAS_BACKEND"] = "torch"  # or "jax", "tensorflow"

from k3_node.datasets import Planetoid
from k3_node.tasks import NodeClassifier

# Load Cora dataset and train a GCN classifier
dataset = Planetoid(root="/tmp/Cora", name="Cora")
data = dataset[0]

clf = NodeClassifier(backbone="gcn", hidden_channels=32, num_layers=2)
clf.fit(data, epochs=30)

# Evaluate
metrics = clf.evaluate(data, mask="test_mask")
print("Accuracy:", metrics["accuracy"])

# Save locally in Hugging Face Hub format
clf.save_pretrained("./my_cora_gcn", metrics=metrics, dataset_name="Cora")
```

This creates a standard directory structure:
```
my_cora_gcn/
├── README.md               # Generated Model Card with tags and metrics
├── config.json             # Model architecture & hyperparameters
└── model.weights.h5        # Keras 3 neural network weights
```

### Loading from Local Directory

```python
# Load using the specific task class
loaded_clf = NodeClassifier.from_pretrained("./my_cora_gcn")

# Or load generically (K3-Node inspects config.json to auto-instantiate the right task!)
from k3_node.hub import from_pretrained
generic_clf = from_pretrained("./my_cora_gcn")

# Run predictions
preds = generic_clf.predict(data)
```

---

## 2. Publishing Models to Hugging Face Hub

Push your trained model directly to your Hugging Face account:

```python
# Authenticate (or run `huggingface-cli login` in your terminal)
# token = "hf_..."

# Push to Hub
repo_url = clf.push_to_hub(
    repo_id="your-username/cora-node-gcn",
    metrics=metrics,
    dataset_name="Cora",
    commit_message="Initial release of trained Cora GCN",
    private=False,
)
print("Model published at:", repo_url)
```

---

## 3. Loading Pretrained Models from the Hub

Anyone can load and run your published model with a single line of code:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"  # Works across all backends!

from k3_node.tasks import NodeClassifier

# Load directly from the Hugging Face Hub
model = NodeClassifier.from_pretrained("your-username/cora-node-gcn")

# Run inference
predictions = model.predict(new_graph)
probabilities = model.predict_proba(new_graph)
```

All 4 high-level task estimators support `save_pretrained`, `from_pretrained`, and `push_to_hub`:
- [`NodeClassifier`](file:///home/anas/k3-node/k3_node/tasks/node_classification.py)
- [`GraphClassifier`](file:///home/anas/k3-node/k3_node/tasks/graph_classification.py)
- [`GraphRegressor`](file:///home/anas/k3-node/k3_node/tasks/graph_regression.py)
- [`LinkPredictor`](file:///home/anas/k3-node/k3_node/tasks/link_prediction.py)

---

## 4. Graph Dataset Hub Integration

Sharing graph datasets (single graphs or collections of graphs) is just as simple:

### Pushing a Graph Dataset to the Hub

```python
from k3_node.hub import push_dataset_to_hub
from k3_node.datasets import TUDataset

# Load a collection of molecular graphs
mutag = TUDataset(root="/tmp/MUTAG", name="MUTAG")

# Push to Hugging Face Hub as a dataset repo
dataset_url = push_dataset_to_hub(
    dataset=list(mutag),
    repo_id="your-username/mutag-graphs",
    description="MUTAG mutagenic aromatic and heteroaromatic nitro compounds graph benchmark.",
)
print("Dataset published at:", dataset_url)
```

### Loading a Graph Dataset from the Hub

```python
from k3_node.hub import load_dataset_from_hub

# Load directly into K3-Node Data structures
graphs = load_dataset_from_hub("your-username/mutag-graphs")
print(f"Loaded {len(graphs)} graphs! Sample: {graphs[0]}")
```

---

## 5. Summary Table

| Operation | Model Hub Function / Method | Dataset Hub Function |
| :--- | :--- | :--- |
| **Save Locally** | `model.save_pretrained("./dir")` | `save_graph_dataset(data, "path.npz")` |
| **Load Locally** | `Task.from_pretrained("./dir")` | `load_graph_dataset("path.npz")` |
| **Push to Hub** | `model.push_to_hub("org/repo")` | `push_dataset_to_hub(data, "org/repo")` |
| **Load from Hub** | `Task.from_pretrained("org/repo")` | `load_dataset_from_hub("org/repo")` |
