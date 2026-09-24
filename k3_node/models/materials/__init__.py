"""Materials and crystal graph neural network models (multi-backend Keras 3 ports of MatGL)."""

from .core import (
    SoftPlus2,
    SoftExponential,
    MLP,
    GatedMLP,
    EmbeddingBlock,
    vector_to_skewtensor,
    vector_to_symtensor,
    decompose_tensor,
    new_radial_tensor,
    tensor_norm,
)
from .basis import (
    GaussianExpansion,
    RadialBesselFunction,
    FourierExpansion,
    ChebyshevRadialBasis,
    SphericalBesselFunction,
    SphericalBesselWithHarmonics,
    BondExpansion,
    polynomial_cutoff,
    cosine_cutoff,
    compute_pair_vector_and_distance,
    compute_theta,
    compute_theta_and_phi,
)
from .readout import (
    ReduceReadOut,
    WeightedReadOut,
    WeightedAtomReadOut,
    Set2SetReadOut,
    EdgeSet2Set,
)
from .megnet import MEGNet, MEGNetBlock, MEGNetGraphConv
from .m3gnet import M3GNet, M3GNetBlock, M3GNetGraphConv, ThreeBodyInteractions
from .tensornet import TensorNet, TensorEmbedding, TensorNetInteraction
from .chgnet import CHGNet, CHGNetAtomGraphBlock, CHGNetBondGraphBlock
from .so3net import SO3Net, SO3Convolution, RealSphericalHarmonics
from .grace import GRACE, GraceSPBasis, GraceACEStack
from .qet import QET, LinearQeq, ElectrostaticPotential
from .wrappers import TransformedTargetModel, Potential
from .io import (
    download_matgl_checkpoint,
    load_matgl_weights,
    load_model,
    get_available_pretrained_models,
)

__all__ = [
    # Models
    "MEGNet",
    "MEGNetBlock",
    "MEGNetGraphConv",
    "M3GNet",
    "M3GNetBlock",
    "M3GNetGraphConv",
    "ThreeBodyInteractions",
    "TensorNet",
    "TensorEmbedding",
    "TensorNetInteraction",
    "CHGNet",
    "CHGNetAtomGraphBlock",
    "CHGNetBondGraphBlock",
    "SO3Net",
    "SO3Convolution",
    "RealSphericalHarmonics",
    "GRACE",
    "GraceSPBasis",
    "GraceACEStack",
    "QET",
    "LinearQeq",
    "ElectrostaticPotential",
    "TransformedTargetModel",
    "Potential",
    # Basis & Geometry
    "GaussianExpansion",
    "RadialBesselFunction",
    "FourierExpansion",
    "ChebyshevRadialBasis",
    "SphericalBesselFunction",
    "SphericalBesselWithHarmonics",
    "BondExpansion",
    "polynomial_cutoff",
    "cosine_cutoff",
    "compute_pair_vector_and_distance",
    "compute_theta",
    "compute_theta_and_phi",
    # Core & Readout
    "SoftPlus2",
    "SoftExponential",
    "MLP",
    "GatedMLP",
    "EmbeddingBlock",
    "vector_to_skewtensor",
    "vector_to_symtensor",
    "decompose_tensor",
    "new_radial_tensor",
    "tensor_norm",
    "ReduceReadOut",
    "WeightedReadOut",
    "WeightedAtomReadOut",
    "Set2SetReadOut",
    "EdgeSet2Set",
    # Checkpoint I/O
    "download_matgl_checkpoint",
    "load_matgl_weights",
    "load_model",
    "get_available_pretrained_models",
]

