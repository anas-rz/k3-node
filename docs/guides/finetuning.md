# Fine-Tuning Recipes

K3-Node ships several families of pretrained models — general graph foundation models ([Pretrained Foundation Models](../api/pretrained.md)), materials-science interatomic potentials ([Materials & Crystal GNNs](../api/materials.md)), and structural-biology docking models ([Biological & Macromolecular Models](../api/bio.md)) — each with a `download_*_checkpoint` / `load_*_weights` pair that loads official pretrained weights directly into the equivalent Keras 3 model. This guide walks through the general fine-tuning pattern, then gives a complete worked example for each domain.

All recipes below run unchanged on the PyTorch, TensorFlow, or JAX backend — just set `KERAS_BACKEND` before importing `keras`/`k3_node`.

---

## The General Pattern

Fine-tuning any K3-Node pretrained model follows the same four steps:

1. **Instantiate** the model with the architecture hyperparameters matching the checkpoint (these are documented on each checkpoint's model card / in the corresponding API page), and with your downstream task's output shape (e.g. `num_tasks=1` for single-target regression).
2. **Download & load** the pretrained weights with the model's `download_*_checkpoint` + `load_*_weights` functions. Loaders match sublayers **by name**, so any task-specific head you added that wasn't part of the original checkpoint (e.g. a new `Dense` output layer) is simply left at its random initialization — the loader silently skips keys it doesn't recognize.
3. **Decide what to freeze.** For small downstream datasets, freeze the backbone (`layer.trainable = False`) and train only the new head first; for larger datasets, unfreeze the backbone afterward and fine-tune everything end-to-end at a lower learning rate.
4. **Compile and train** exactly as you would any Keras model — `model.compile(...)` + `model.fit(...)` or a manual `train_on_batch` loop.

```python
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

# 1. Instantiate with the downstream task's output shape
model = MyPretrainedModel(**pretrained_config, num_tasks=1)

# 2. Load pretrained weights (task head stays randomly initialized)
ckpt = download_my_checkpoint("some-checkpoint-name")
load_my_weights(model, ckpt)

# 3. Freeze the backbone for a warm-up phase
model.backbone.trainable = False

# 4. Compile & train the head
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
model.fit(train_data, epochs=5)

# Optionally unfreeze and fine-tune end-to-end at a lower LR
model.backbone.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="mse")
model.fit(train_data, epochs=20)
```

---

## Molecular Property Fine-Tuning (Mole-BERT)

[`MoleBERT`](../api/pretrained.md#5-mole-bert-self-supervised-molecular-gnn) pretrains a 5-layer GIN backbone via masked atom/bond modeling; its constructor accepts `num_tasks` directly, so the downstream head is built in from the start — `load_mole_bert_weights` only ever touches the GIN backbone (`model.gnn`), never the head, so this works with any `num_tasks` value.

```python
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from k3_node.models.mole_bert import MoleBERT, load_mole_bert_weights, download_mole_bert_checkpoint
from k3_node.datasets import MoleculeNet
from k3_node.loader import DataLoader

# 1. Instantiate with your downstream task's output size (e.g. 1 regression target)
model = MoleBERT(num_layer=5, emb_dim=300, num_tasks=1, graph_pooling="mean")
model.build(None)

# 2. Load the pretrained backbone (graph_pred_linear head stays freshly initialized)
ckpt_path = download_mole_bert_checkpoint()
load_mole_bert_weights(model, ckpt_path)

# 3. Freeze the backbone for a short warm-up on the new head
model.gnn.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.MeanSquaredError())

dataset = MoleculeNet(root="./data/MoleculeNet", name="ESOL")
train_loader = DataLoader(dataset[:900], batch_size=32, shuffle=True, drop_last=True)

def make_generator(loader, batch_size):
    while True:
        for batch in loader:
            inputs = (batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            yield inputs, batch.y

model.fit(make_generator(train_loader, 32), steps_per_epoch=len(train_loader), epochs=5)

# 4. Unfreeze and fine-tune end-to-end at a lower LR
model.gnn.trainable = True
model.compile(optimizer=keras.optimizers.Adam(1e-5), loss=keras.losses.MeanSquaredError())
model.fit(make_generator(train_loader, 32), steps_per_epoch=len(train_loader), epochs=20)
```

!!! tip "Static batch size for graph-level readouts"
    Notice `drop_last=True` on the training `DataLoader`, keeping every batch at exactly 32 graphs. Any model that reads out a graph-level prediction via `global_add_pool`/`global_mean_pool`/`global_max_pool` should pass an explicit, **statically known** `size=batch_size` to the pool call (or otherwise guarantee a fixed graph count per batch) rather than relying on `batch.max() + 1`. Keras's `fit()`/`train_on_batch()` runs one internal validation pass with constant-filled placeholder tensors before real training starts; if the pooled output size depends on the *values* inside `batch` rather than being a fixed constant, that placeholder pass infers the wrong size and raises a spurious shape-mismatch error. `MoleBERT`'s own pooling already takes a `batch` tensor that's dense per training batch, but if you build a custom readout on top of a pretrained backbone, keep this in mind — see [`colors_topk_pool.ipynb`](https://github.com/anas-rz/k3-node/blob/main/examples/colors_topk_pool.ipynb) or [`mutag_gin.ipynb`](https://github.com/anas-rz/k3-node/blob/main/examples/mutag_gin.ipynb) for the fixed pattern.

---

## Materials Property Fine-Tuning (M3GNet / MEGNet)

The [materials models](../api/materials.md) are typically distributed as a `TransformedTargetModel`-wrapped checkpoint (predictions are denormalized by a stored `mean`/`std`). `load_model` is the fastest path to a ready-to-use pretrained model; for fine-tuning on a *different* property than the checkpoint was trained for, unwrap `.model` to reach the raw backbone and attach it to a fresh readout.

```python
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from k3_node.models.materials import load_model, M3GNet

# Fine-tune directly on the same property (e.g. domain-shifted formation energy data)
pretrained = load_model("M3GNet-Eform-MP-2018.6.1")   # TransformedTargetModel(M3GNet(...))
backbone = pretrained.model                            # the raw M3GNet, already pretrained
backbone.trainable = True

pretrained.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.MeanAbsoluteError())
pretrained.fit(train_crystal_graphs, train_targets, epochs=50, batch_size=16)
```

```python
# Fine-tune on a *different* scalar property: reuse the pretrained backbone's
# representation but attach a fresh readout instead of the checkpoint's head.
from k3_node.models.materials.wrappers import TransformedTargetModel

backbone = pretrained.model
new_model = TransformedTargetModel(backbone, mean=my_target_mean, std=my_target_std)
new_model.compile(optimizer=keras.optimizers.Adam(1e-4), loss=keras.losses.MeanAbsoluteError())
new_model.fit(train_crystal_graphs, train_targets, epochs=50, batch_size=16)
```

Recall from the [Materials API page](../api/materials.md) that every crystal-graph input is a `dict` with `pos`, `edge_index`, `node_type`, `batch`, and (for 3-body models like `M3GNet`/`CHGNet`/`GRACE`) `line_edge_index` keys — build these with your own crystal-to-graph preprocessing, or reuse `k3_node.transforms` utilities that already produce this shape for the bundled datasets.

For potential (force/stress-aware) fine-tuning rather than a single scalar property, wrap the backbone in [`Potential`](../api/materials.md#potential) instead of `TransformedTargetModel`, and enable `calc_forces=True`.

---

## Node/Graph Representation Fine-Tuning (GraphMAE2)

[`GraphMAE2`](../api/pretrained.md#1-graphmae2-masked-autoencoder-for-graphs) is a self-supervised masked autoencoder — after pretraining, only `model.encoder` is useful downstream (the decoder reconstructs masked features and is discarded). Attach a new task head directly to the encoder's output embeddings.

```python
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import layers, ops
from k3_node.models.graphmae2 import GraphMAE2, load_graphmae2_weights, download_graphmae2_checkpoint

pretrained = GraphMAE2(in_dim=128, num_hidden=512, num_layers=2, nhead=4, nhead_out=1)
pretrained.build(None)

ckpt_path = download_graphmae2_checkpoint("ogbn-arxiv")
load_graphmae2_weights(pretrained, ckpt_path)

class NodeClassifier(keras.Model):
    def __init__(self, encoder, hidden_dim, num_classes):
        super().__init__()
        self.encoder = encoder          # pretrained, frozen or fine-tuned
        self.head = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        x, edge_index = inputs
        h = self.encoder(x, edge_index, training=training)
        return self.head(h)

model = NodeClassifier(pretrained.encoder, hidden_dim=512, num_classes=40)
model.encoder.trainable = False  # linear-probe the frozen representation first

model.compile(
    optimizer=keras.optimizers.Adam(1e-2),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
)
model.fit((node_features, edge_index), labels, epochs=100)
```

The same pattern applies to `Graphormer`/`GraphGPS`/`GROVER` — instantiate the model, load pretrained weights, then either read `model(inputs, features_only=True)`-style representations (where supported) or graft a new head onto the model's pooled output.

---

## Structural Biology Fine-Tuning (Uni-Mol Docking)

[`UniMolDockingModel`](../api/bio.md#unimoldockingmodel) is already end-to-end (it directly predicts a pose update and pairwise distances, no separate head to attach), so fine-tuning is a matter of loading the pretrained backbone and continuing training on your own protein–ligand complexes at a low learning rate:

```python
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from k3_node.models.unimol import UniMolDockingModel, download_unimol_checkpoint, load_unimol_weights

model = UniMolDockingModel(output_dim=2, data_type="molecule")
ckpt = download_unimol_checkpoint("binding_pose")
load_unimol_weights(model, checkpoint_path=ckpt)

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # small LR: full end-to-end fine-tune
    loss={"pose": keras.losses.MeanSquaredError(), "dist": keras.losses.MeanSquaredError()},
)
model.fit(train_complexes, epochs=10)
```

For a lighter-touch adaptation, freeze `model.encoder` and only fine-tune `model.docking_coord_head` and `model.dist_head` initially, then unfreeze the encoder for a final low-LR pass — the same warm-up/unfreeze pattern used in the molecular property example above.

---

## Common Gotchas

- **Checkpoint hyperparameters must match exactly.** `download_*_checkpoint`/`load_*_weights` match weights by name and shape; if you change a hidden dimension, layer count, or head count from what the checkpoint was trained with, loading will silently skip mismatched tensors (or raise a shape error) rather than partially resize them. Check each model's page for the exact config the official checkpoint expects.
- **New heads are never touched by the loader.** This is a feature (it's what makes `num_tasks=<your value>` "just work" with a pretrained checkpoint), but it also means a freshly-attached head starts randomly initialized every time — always run at least a short warm-up phase with the backbone frozen before unfreezing everything, especially with a small fine-tuning dataset.
- **Fixed batch sizes for graph-level pooling.** As called out above, any custom readout built with `global_*_pool` should receive a static `size` when used inside `model.fit()`/`train_on_batch()`, to avoid Keras's build-time validation pass inferring the wrong output shape from a placeholder batch.
- **Backend parity.** All pretrained checkpoints are distributed as PyTorch state dicts and converted into the current Keras backend's weight format on load, so switching `KERAS_BACKEND` between fine-tuning runs is safe — just re-run `load_*_weights` after switching, since weights aren't portable across a live Python process's backend switch.
