# Materials & Crystal Graph Neural Networks

The `k3_node.materials` module (backed by `k3_node.models.materials`) provides multi-backend Keras 3 ports of the [MatGL](https://github.com/materialsvirtuallab/matgl) family of interatomic potentials and crystal property predictors, plus built-in checkpoint downloaders that load official pretrained PyTorch weights (hosted on the Hugging Face Hub under the `materialyze` organization) directly into the Keras models.

All models consume a **crystal graph** input — a `dict` (or `k3_node.data.Data`-like object) with the following keys:

| Key | Shape | Description |
|---|---|---|
| `pos` | `[N, 3]` | Atomic Cartesian coordinates. |
| `edge_index` | `[2, E]` | Bonded-neighbor pairs within the model's cutoff radius. |
| `node_type` | `[N]` | Atomic numbers (or a mapped element index). |
| `batch` | `[N]` | Graph id per node, for batched crystals. |
| `line_edge_index` | `[2, L]` | Bond-pair (angle) connectivity, required by 3-body models (`M3GNet`, `CHGNet`, `GRACE`). |
| `state_attr` | `[num_graphs, S]` | Optional global/state features (e.g. temperature), used by `MEGNet`. |

Every model outputs a single scalar per crystal (shape `()`/`(1,)` for a single graph, `(num_graphs,)` when batched) — typically formation energy, band gap, or another crystal-level property.

---

## Models

### MEGNet
[MEGNet](https://arxiv.org/abs/1812.05055): a graph network with dedicated node/edge/global ("state") update blocks, alternating graph convolution with global-state pooling and broadcasting.

::: k3_node.models.materials.megnet.MEGNet

**Usage:**
```python
from k3_node.models.materials import MEGNet

model = MEGNet(
    dim_node_embedding=16,
    dim_edge_embedding=20,
    dim_state_embedding=2,
    nblocks=2,
    hidden_layer_sizes_input=(32, 16),
    hidden_layer_sizes_conv=(32, 16),
    hidden_layer_sizes_output=(16,),
)
out = model(crystal_graph)  # scalar property prediction
```

### M3GNet
[M3GNet](https://arxiv.org/abs/2202.02450): a materials 3-body graph network combining 2-body (bond) and 3-body (angle) interactions for interatomic potential and property prediction, used as MatGL's flagship universal potential.

::: k3_node.models.materials.m3gnet.M3GNet

### TensorNet
[TensorNet](https://arxiv.org/abs/2306.06482): a Cartesian-tensor message-passing network that represents edge features as rank-0/1/2 tensors (scalar, vector, and symmetric-traceless tensor channels) for equivariant potential prediction.

::: k3_node.models.materials.tensornet.TensorNet

### CHGNet
[CHGNet](https://arxiv.org/abs/2302.14231): a charge-informed 3-body graph network that additionally predicts atomic magnetic moments, useful for magnetism-aware materials properties.

::: k3_node.models.materials.chgnet.CHGNet

### SO3Net
An SO(3)-equivariant network built on real spherical harmonics convolutions, for rotation-equivariant potential prediction.

::: k3_node.models.materials.so3net.SO3Net

### GRACE
A graph atomic cluster expansion (ACE)-style network combining a spherical-harmonic product basis with tensor-product message passing.

::: k3_node.models.materials.grace.GRACE

### QET
A charge-equilibration-aware potential (couples a `LinearQeq` electronegativity-equilibration module with an `ElectrostaticPotential` energy term) for systems where long-range electrostatics matter.

::: k3_node.models.materials.qet.QET

---

## Model Wrappers

### Potential
Wraps any of the above energy models as an interatomic potential, denormalizing predictions with `data_mean`/`data_std` and (optionally) computing per-atom forces via autodiff.

::: k3_node.models.materials.wrappers.Potential

### TransformedTargetModel
A thin wrapper that applies `pred * std + mean` to a model's output — this is what pretrained MatGL checkpoints are usually distributed as, since targets are normalized during training.

::: k3_node.models.materials.wrappers.TransformedTargetModel

---

## Pretrained Checkpoints

### load_model
The recommended one-call entry point: downloads (if needed), parses the MatGL `model.json` architecture spec, instantiates the matching model class with the right constructor arguments, and loads the pretrained `state.pt` weights — wrapping the result in a `TransformedTargetModel` when the checkpoint was trained on normalized targets.

::: k3_node.models.materials.io.load_model

**Usage:**
```python
from k3_node.models.materials import load_model, get_available_pretrained_models

print(get_available_pretrained_models())
# e.g. ['CHGNet-PES-MatPES-PBE-2025.2.10', 'M3GNet-Eform-MP-2018.6.1', ...]

model = load_model("M3GNet-Eform-MP-2018.6.1")
formation_energy = model(crystal_graph)
```

### get_available_pretrained_models
Lists the pretrained checkpoints published under the `materialyze` Hugging Face organization (falls back to a hardcoded list if offline).

::: k3_node.models.materials.io.get_available_pretrained_models

### download_matgl_checkpoint
Downloads a checkpoint's `model.json` (architecture) and `state.pt` (weights) files from the Hugging Face Hub without instantiating a model — use this if you want to inspect or manually load a checkpoint.

::: k3_node.models.materials.io.download_matgl_checkpoint

### load_matgl_weights
Lower-level weight loader: copies a PyTorch `state_dict` (or a path to one) into an already-constructed Keras model, matching sublayers by name. Prefer `load_model` unless you need to load weights into a model you built yourself (e.g. for fine-tuning with a different output head — see the [fine-tuning recipes](../guides/finetuning.md#materials-property-fine-tuning-m3gnet-megnet)).

::: k3_node.models.materials.io.load_matgl_weights

---

## Building Blocks

Lower-level components used internally by the models above — useful when assembling a custom materials architecture.

### Radial & Angular Basis Functions
::: k3_node.models.materials.basis.GaussianExpansion
::: k3_node.models.materials.basis.BondExpansion
::: k3_node.models.materials.basis.RadialBesselFunction
::: k3_node.models.materials.basis.SphericalBesselFunction
::: k3_node.models.materials.basis.SphericalBesselWithHarmonics
::: k3_node.models.materials.basis.FourierExpansion
::: k3_node.models.materials.basis.ChebyshevRadialBasis

### Geometry Utilities
::: k3_node.models.materials.basis.compute_pair_vector_and_distance
::: k3_node.models.materials.basis.compute_theta
::: k3_node.models.materials.basis.compute_theta_and_phi
::: k3_node.models.materials.basis.polynomial_cutoff
::: k3_node.models.materials.basis.cosine_cutoff

### Core Layers
::: k3_node.models.materials.core.EmbeddingBlock
::: k3_node.models.materials.core.MLP
::: k3_node.models.materials.core.GatedMLP
::: k3_node.models.materials.core.SoftPlus2
::: k3_node.models.materials.core.SoftExponential

### Tensor Utilities (TensorNet)
::: k3_node.models.materials.core.vector_to_skewtensor
::: k3_node.models.materials.core.vector_to_symtensor
::: k3_node.models.materials.core.decompose_tensor
::: k3_node.models.materials.core.new_radial_tensor
::: k3_node.models.materials.core.tensor_norm

### Readout Heads
::: k3_node.models.materials.readout.ReduceReadOut
::: k3_node.models.materials.readout.WeightedReadOut
::: k3_node.models.materials.readout.WeightedAtomReadOut
::: k3_node.models.materials.readout.Set2SetReadOut
::: k3_node.models.materials.readout.EdgeSet2Set
