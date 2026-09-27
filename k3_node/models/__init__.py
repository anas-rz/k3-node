r"""k3-node ports of `torch_geometric.nn.models`."""

from .mlp import MLP
from .attract_repel import ARLinkPredictor
from .autoencoder import InnerProductDecoder, GAE, VGAE, ARGA, ARGVA
from .deep_graph_infomax import DeepGraphInfomax
from .deepgcn import DeepGCNLayer
from .attentive_fp import AttentiveFP
from .jumping_knowledge import JumpingKnowledge, HeteroJumpingKnowledge
from .mask_label import MaskLabel
from .meta import MetaLayer
from .pmlp import PMLP
from .polynormer import Polynormer
from .basic_gnn import BasicGNN, GCN, GraphSAGE, GIN, GAT, PNA, EdgeCNN
from .label_prop import LabelPropagation
from .correct_and_smooth import CorrectAndSmooth
from .lightgcn import LightGCN, BPRLoss
from .linkx import LINKX, SparseLinear
from .rect import RECT_L
from .signed_gcn import SignedGCN
from .neural_fingerprint import NeuralFingerprint
from .graph_unet import GraphUNet
from .rev_gnn import GroupAddRev
from .sgformer import SGFormer
from .node2vec import Node2Vec
from .metapath2vec import MetaPath2Vec
from .renet import RENet
from .tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
    MeanAggregator,
    TimeEncoder,
    LastNeighborLoader,
)
from .schnet import (
    SchNet,
    CFConv,
    InteractionBlock as SchNetInteractionBlock,
    GaussianSmearing,
    ShiftedSoftplus,
    RadiusInteractionGraph,
)
from .dimenet import (
    DimeNet,
    DimeNetPlusPlus,
    BesselBasisLayer,
    SphericalBasisLayer,
    triplets,
)
from .gnnff import (
    GNNFF,
    NodeBlock,
    EdgeBlock,
    GaussianFilter,
)
from .gpse import (
    GPSE,
    GPSENodeEncoder,
    GeneralLayer,
    GeneralMultiLayer,
    GNNStackStage,
    GNNInductiveHybridMultiHead,
)
from .visnet import ViSNet
from .lpformer import LPFormer, LPAttLayer
from .graphmae2 import (
    GraphMAE2,
    sce_loss,
    load_graphmae2_weights,
    download_graphmae2_checkpoint,
)
from .graphormer import (
    Graphormer,
    GraphNodeFeature,
    GraphAttnBias,
    GraphormerMultiheadAttention,
    GraphormerGraphEncoderLayer,
    GraphormerGraphEncoder,
    load_graphormer_weights,
    download_graphormer_checkpoint,
)
from .graphormer_3d import (
    Graphormer3D,
    GaussianLayer,
    RBF,
    Graphormer3DEncoderLayer,
    NodeTaskHead,
    load_graphormer3d_weights,
    download_graphormer3d_checkpoint,
)
from .captum import to_captum_model, to_captum_input, captum_output_to_dicts
from .gps_model import (
    GPSModel,
    GPSLayer,
    CustomGatedGCN,
    AtomEncoder,
    BondEncoder,
    RWSEEncoder,
    SANGraphHead,
    load_gps_weights,
    download_gps_checkpoint,
)
from .grover import (
    GROVER,
    GTransEncoder,
    Readout,
    load_grover_weights,
    download_grover_checkpoint,
)
from .mole_bert import (
    MoleBERT,
    MoleBERTGNN,
    MoleBERTGINConv,
    load_mole_bert_weights,
    download_mole_bert_checkpoint,
)
from .unimol import (
    UniMolModel,
    UniMolConfGenModel,
    UniMolDockingModel,
    GaussianLayer as UniMolGaussianLayer,
    NumericalEmbed as UniMolNumericalEmbed,
    NonLinearHead as UniMolNonLinearHead,
    DistanceHead as UniMolDistanceHead,
    ClassificationHead as UniMolClassificationHead,
    LinearHead as UniMolLinearHead,
    MaskLMHead as UniMolMaskLMHead,
    download_unimol_checkpoint,
    load_unimol_weights,
)
from .unimol2 import (
    UniMol2Model,
    AtomFeature as UniMol2AtomFeature,
    EdgeFeature as UniMol2EdgeFeature,
    SE3InvariantKernel as UniMol2SE3Kernel,
    MovementPredictionHead as UniMol2MovementHead,
    download_unimol2_checkpoint,
    load_unimol2_weights,
)
from .unimol_plus import (
    UniMolPlusPCQModel,
    UniMolPlusOC20Model,
    EnergyHead as UniMolPlusEnergyHead,
    download_unimol_plus_checkpoint,
    load_unimol_plus_weights,
)
from .unimol_docking_v2 import (
    DockingPoseModelV2,
    download_unimol_docking_checkpoint,
    load_unimol_docking_weights,
)
from . import materials
from . import bio
from . import chemistry
from .materials import (
    MEGNet,
    MEGNetBlock,
    MEGNetGraphConv,
    M3GNet,
    M3GNetBlock,
    M3GNetGraphConv,
    ThreeBodyInteractions,
    TensorNet,
    TensorEmbedding,
    TensorNetInteraction,
    CHGNet,
    CHGNetAtomGraphBlock,
    CHGNetBondGraphBlock,
    SO3Net,
    SO3Convolution,
    RealSphericalHarmonics,
    GRACE,
    GraceSPBasis,
    GraceACEStack,
    QET,
    LinearQeq,
    ElectrostaticPotential,
    TransformedTargetModel,
    Potential,
    BondExpansion as MatGLBondExpansion,
    GaussianExpansion as MatGLGaussianExpansion,
    RadialBesselFunction as MatGLRadialBesselFunction,
    FourierExpansion as MatGLFourierExpansion,
    ChebyshevRadialBasis as MatGLChebyshevRadialBasis,
    SphericalBesselFunction as MatGLSphericalBesselFunction,
    SphericalBesselWithHarmonics as MatGLSphericalBesselWithHarmonics,
    ReduceReadOut as MatGLReduceReadOut,
    WeightedReadOut as MatGLWeightedReadOut,
    WeightedAtomReadOut as MatGLWeightedAtomReadOut,
    Set2SetReadOut as MatGLSet2SetReadOut,
    EdgeSet2Set as MatGLEdgeSet2Set,
    download_matgl_checkpoint,
    load_matgl_weights,
    load_model as load_matgl_model,
    get_available_pretrained_models as get_available_matgl_models,
)

__all__ = [
    "materials",
    "bio",
    "chemistry",
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
    "MatGLBondExpansion",
    "MatGLGaussianExpansion",
    "MatGLRadialBesselFunction",
    "MatGLFourierExpansion",
    "MatGLChebyshevRadialBasis",
    "MatGLSphericalBesselFunction",
    "MatGLSphericalBesselWithHarmonics",
    "MatGLReduceReadOut",
    "MatGLWeightedReadOut",
    "MatGLWeightedAtomReadOut",
    "MatGLSet2SetReadOut",
    "MatGLEdgeSet2Set",
    "download_matgl_checkpoint",
    "load_matgl_weights",
    "load_matgl_model",
    "get_available_matgl_models",
    "MLP",
    "ARLinkPredictor",
    "InnerProductDecoder",
    "GAE",
    "VGAE",
    "ARGA",
    "ARGVA",
    "DeepGraphInfomax",
    "DeepGCNLayer",
    "AttentiveFP",
    "JumpingKnowledge",
    "HeteroJumpingKnowledge",
    "MaskLabel",
    "MetaLayer",
    "PMLP",
    "Polynormer",
    "BasicGNN",
    "GCN",
    "GraphSAGE",
    "GIN",
    "GAT",
    "PNA",
    "EdgeCNN",
    "LabelPropagation",
    "CorrectAndSmooth",
    "LightGCN",
    "BPRLoss",
    "LINKX",
    "SparseLinear",
    "RECT_L",
    "SignedGCN",
    "NeuralFingerprint",
    "GraphUNet",
    "GroupAddRev",
    "SGFormer",
    "Node2Vec",
    "MetaPath2Vec",
    "RENet",
    "TGNMemory",
    "IdentityMessage",
    "LastAggregator",
    "MeanAggregator",
    "TimeEncoder",
    "LastNeighborLoader",
    "SchNet",
    "CFConv",
    "SchNetInteractionBlock",
    "GaussianSmearing",
    "ShiftedSoftplus",
    "RadiusInteractionGraph",
    "DimeNet",
    "DimeNetPlusPlus",
    "BesselBasisLayer",
    "SphericalBasisLayer",
    "triplets",
    "GNNFF",
    "NodeBlock",
    "EdgeBlock",
    "GaussianFilter",
    "GPSE",
    "GPSENodeEncoder",
    "GeneralLayer",
    "GeneralMultiLayer",
    "GNNStackStage",
    "GNNInductiveHybridMultiHead",
    "ViSNet",
    "LPFormer",
    "LPAttLayer",
    "GraphMAE2",
    "sce_loss",
    "load_graphmae2_weights",
    "download_graphmae2_checkpoint",
    "Graphormer",
    "GraphNodeFeature",
    "GraphAttnBias",
    "GraphormerMultiheadAttention",
    "GraphormerGraphEncoderLayer",
    "GraphormerGraphEncoder",
    "load_graphormer_weights",
    "download_graphormer_checkpoint",
    "Graphormer3D",
    "GaussianLayer",
    "RBF",
    "Graphormer3DEncoderLayer",
    "NodeTaskHead",
    "load_graphormer3d_weights",
    "download_graphormer3d_checkpoint",
    "to_captum_model",
    "to_captum_input",
    "captum_output_to_dicts",
    "GPSModel",
    "GPSLayer",
    "CustomGatedGCN",
    "AtomEncoder",
    "BondEncoder",
    "RWSEEncoder",
    "SANGraphHead",
    "load_gps_weights",
    "download_gps_checkpoint",
    "GROVER",
    "GTransEncoder",
    "Readout",
    "load_grover_weights",
    "download_grover_checkpoint",
    "MoleBERT",
    "MoleBERTGNN",
    "MoleBERTGINConv",
    "load_mole_bert_weights",
    "download_mole_bert_checkpoint",
    "UniMolModel",
    "UniMolConfGenModel",
    "UniMolDockingModel",
    "UniMolGaussianLayer",
    "UniMolNumericalEmbed",
    "UniMolNonLinearHead",
    "UniMolDistanceHead",
    "UniMolClassificationHead",
    "UniMolLinearHead",
    "UniMolMaskLMHead",
    "download_unimol_checkpoint",
    "load_unimol_weights",
    "UniMol2Model",
    "UniMol2AtomFeature",
    "UniMol2EdgeFeature",
    "UniMol2SE3Kernel",
    "UniMol2MovementHead",
    "download_unimol2_checkpoint",
    "load_unimol2_weights",
    "UniMolPlusPCQModel",
    "UniMolPlusOC20Model",
    "UniMolPlusEnergyHead",
    "download_unimol_plus_checkpoint",
    "load_unimol_plus_weights",
    "DockingPoseModelV2",
    "download_unimol_docking_checkpoint",
    "load_unimol_docking_weights",
]

# Inject Hugging Face Hub capabilities (from_pretrained, save_pretrained, push_to_hub, predict)
# to all models in k3_node.models
import keras
from k3_node.hub.hub_mixin import K3NodeHubMixin

def _defines_own(cls, attr):
    """True if a k3_node class in ``cls``'s MRO defines ``attr`` itself (e.g. a model-specific
    ``from_pretrained`` that loads original checkpoints), which must not be overwritten."""
    return any(attr in vars(klass) for klass in cls.__mro__ if klass.__module__.startswith("k3_node"))


def _is_graph_input(data):
    if hasattr(data, "edge_index") or (hasattr(data, "z") and hasattr(data, "pos")):
        return True
    return isinstance(data, dict) and any(k in data for k in ("edge_index", "pos", "z"))


def _graph_aware_predict(self, data=None, *args, **kwargs):
    """Graph inputs (``Data``, ``Batch``, graph dicts) use the hub-style ``predict``; everything
    else keeps Keras' batched ``Model.predict`` (arrays, ``tf.data``, ``PyDataset``, ...)."""
    if _is_graph_input(data):
        return K3NodeHubMixin.predict(self, data, *args, **kwargs)
    return keras.Model.predict(self, data, *args, **kwargs)


for _name in list(__all__):
    _obj = globals().get(_name)
    if isinstance(_obj, type) and issubclass(_obj, (keras.Model, keras.layers.Layer)):
        if not issubclass(_obj, K3NodeHubMixin):
            if not _defines_own(_obj, "from_pretrained"):
                _obj.from_pretrained = classmethod(K3NodeHubMixin.from_pretrained.__func__)
            if not _defines_own(_obj, "save_pretrained"):
                _obj.save_pretrained = K3NodeHubMixin.save_pretrained
            if not _defines_own(_obj, "push_to_hub"):
                _obj.push_to_hub = K3NodeHubMixin.push_to_hub
            if not _defines_own(_obj, "predict"):
                if issubclass(_obj, keras.Model):
                    _obj.predict = _graph_aware_predict
                else:
                    _obj.predict = K3NodeHubMixin.predict
            if not hasattr(_obj, "_get_config"):
                _obj._get_config = K3NodeHubMixin._get_config


