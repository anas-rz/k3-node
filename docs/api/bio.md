# Biological & Macromolecular Models

The `k3_node.bio` module (backed by `k3_node.models`) provides multi-backend Keras 3 ports of Uni-Mol's protein–ligand **binding-pose prediction** models, for docking a small-molecule ligand into a protein pocket given 3D structural input.

---

## Models

### UniMolDockingModel
The original [Uni-Mol](https://openreview.net/forum?id=6K2RM6wVqKu) docking head: a `UniMolModel` subclass that adds a cross-attention coordinate-update head predicting an iterative pose correction for the ligand, alongside a pairwise distance head.

::: k3_node.models.unimol.UniMolDockingModel

**Usage:**
```python
from k3_node.models.unimol import UniMolDockingModel, download_unimol_checkpoint, load_unimol_weights

model = UniMolDockingModel(output_dim=2, data_type="molecule")
ckpt = download_unimol_checkpoint("binding_pose")
load_unimol_weights(model, checkpoint_path=ckpt)

# src_tokens: [batch, seq_len] combined ligand+pocket tokens; src_coord: [batch, seq_len, 3]
pose_update, pred_dist = model(src_tokens, src_coord=src_coord)
```

### DockingPoseModelV2
Uni-Mol **Docking V2**: a joint ligand/pocket transformer (separate token vocabularies and embeddings for molecule vs. pocket, fused through shared pair-aware transformer layers) that predicts both the ligand binding pose and pairwise distance matrix in a single forward pass.

::: k3_node.models.unimol_docking_v2.DockingPoseModelV2

**Usage:**
```python
from k3_node.models.unimol_docking_v2 import (
    DockingPoseModelV2,
    download_unimol_docking_checkpoint,
    load_unimol_docking_weights,
)

model = DockingPoseModelV2(embed_dim=512, pair_dim=128, num_layers=12, num_heads=32)
ckpt = download_unimol_docking_checkpoint()
load_unimol_docking_weights(model, checkpoint_path=ckpt)

pose, pair_dist = model(
    mol_tokens, pocket_tokens=pocket_tokens,
    mol_coords=mol_coords, pocket_coords=pocket_coords,
)
```

---

## Pretrained Checkpoints

### download_unimol_checkpoint
::: k3_node.models.unimol.download_unimol_checkpoint

### load_unimol_weights
::: k3_node.models.unimol.load_unimol_weights

### download_unimol_docking_checkpoint
::: k3_node.models.unimol_docking_v2.download_unimol_docking_checkpoint

### load_unimol_docking_weights
::: k3_node.models.unimol_docking_v2.load_unimol_docking_weights

See the [Fine-Tuning Recipes](../guides/finetuning.md) guide for a worked example of fine-tuning `UniMolDockingModel` on a custom docking dataset.
