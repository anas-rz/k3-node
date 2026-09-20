# Pretrained Foundation Models

K3 Node provides native implementations of modern Graph Foundation Models with built-in checkpoint downloaders and weight loaders for official pre-trained models.

---

## 1. GraphMAE2 (Masked Autoencoder for Graphs)

GraphMAE2 is an advanced masked autoencoder for self-supervised graph representation learning featuring multi-task reconstruction (scaled cosine error) and projection heads.

### Architecture
::: k3_node.models.graphmae2.GraphMAE2

### Checkpoint Loading
::: k3_node.models.graphmae2.load_graphmae2_weights

### Checkpoint Downloader
::: k3_node.models.graphmae2.download_graphmae2_checkpoint

### Usage Example
```python
from k3_node.models.graphmae2 import GraphMAE2, load_graphmae2_weights, download_graphmae2_checkpoint

# Download checkpoint
ckpt_path = download_graphmae2_checkpoint("cora")

# Initialize and load model
model = GraphMAE2(in_dim=1433, num_hidden=512, out_dim=1433, num_layers=2)
load_graphmae2_weights(model, ckpt_path)
```

---

## 2. Graphormer & Graphormer3D

Graphormer is a Graph Transformer with spatial, degree, and edge attention biases. Graphormer3D extends this architecture with 3D Gaussian RBF distance biases for molecular conformation and quantum chemistry.

### Graphormer (2D)
::: k3_node.models.graphormer.Graphormer

### Graphormer3D (3D)
::: k3_node.models.graphormer_3d.Graphormer3D

### Checkpoint Utilities
::: k3_node.models.graphormer.load_graphormer_weights
::: k3_node.models.graphormer.download_graphormer_checkpoint

---

## 3. GraphGPS (General Powerful Scalable Graph Transformer)

GraphGPS is a hybrid graph transformer architecture combining local message-passing neural networks (CustomGatedGCN) and global linear/full attention with Random Walk Structural Encodings (RWSE).

### GPSModel
::: k3_node.models.gps_model.GPSModel

### Checkpoint Utilities
::: k3_node.models.gps_model.load_gps_weights
::: k3_node.models.gps_model.download_gps_checkpoint

### Usage Example
```python
from k3_node.models.gps_model import GPSModel, load_gps_weights, download_gps_checkpoint

# Download official pcqm4m-GPS+RWSE.deep checkpoint
ckpt_path = download_gps_checkpoint("pcqm4m-GPS+RWSE.deep")

# Initialize and load weights
model = GPSModel(dim_in=256, dim_out=1, num_layers=16, dim_hidden=256, num_heads=8)
load_gps_weights(model, ckpt_path)
```

---

## 4. GROVER (Graph Representation from Self-Supervised Message Passing Transformer)

GROVER incorporates dual-track message passing across directed bonds and atoms with transformer self-attention and scope-based molecular readouts.

### GROVER
::: k3_node.models.grover.GROVER

### Checkpoint Utilities
::: k3_node.models.grover.load_grover_weights
::: k3_node.models.grover.download_grover_checkpoint

### Usage Example
```python
from k3_node.models.grover import GROVER, load_grover_weights, download_grover_checkpoint

ckpt_path = download_grover_checkpoint("grover_base")

model = GROVER(hidden_size=800, num_attn_head=4, depth=6, num_mt_block=1)
load_grover_weights(model, ckpt_path)
```

---

## 5. Mole-BERT (Self-Supervised Molecular GNN)

Mole-BERT pre-trains a 5-layer GIN backbone with categorical atom and bond embeddings, batch normalization, and Jumping Knowledge for downstream molecular property prediction.

### MoleBERT
::: k3_node.models.mole_bert.MoleBERT

### Checkpoint Utilities
::: k3_node.models.mole_bert.load_mole_bert_weights
::: k3_node.models.mole_bert.download_mole_bert_checkpoint

### Usage Example
```python
from k3_node.models.mole_bert import MoleBERT, load_mole_bert_weights, download_mole_bert_checkpoint

ckpt_path = download_mole_bert_checkpoint()

model = MoleBERT(num_layer=5, emb_dim=300, num_tasks=1)
load_mole_bert_weights(model, ckpt_path)
```

