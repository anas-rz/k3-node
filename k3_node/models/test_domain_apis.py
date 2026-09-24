"""Test domain-specific API organization (materials, bio, chemistry)."""

import pytest


def test_materials_api():
    # Via k3_node.models.materials
    from k3_node.models.materials import (
        MEGNet,
        M3GNet,
        TensorNet,
        CHGNet,
        SO3Net,
        GRACE,
        QET,
        TransformedTargetModel,
        Potential,
        download_matgl_checkpoint,
        load_matgl_weights,
        load_model,
        get_available_pretrained_models,
    )
    assert MEGNet is not None
    assert M3GNet is not None
    assert TensorNet is not None
    assert CHGNet is not None
    assert SO3Net is not None
    assert GRACE is not None
    assert QET is not None

    # Via top-level k3_node.materials
    from k3_node.materials import (
        MEGNet as MatMEGNet,
        M3GNet as MatM3GNet,
        TensorNet as MatTensorNet,
    )
    assert MatMEGNet is MEGNet
    assert MatM3GNet is M3GNet
    assert MatTensorNet is TensorNet


def test_bio_api():
    # Via k3_node.models.bio
    from k3_node.models.bio import (
        UniMolDockingModel,
        DockingPoseModelV2,
        download_unimol_docking_checkpoint,
        load_unimol_docking_weights,
    )
    assert UniMolDockingModel is not None
    assert DockingPoseModelV2 is not None

    # Via top-level k3_node.bio
    from k3_node.bio import (
        UniMolDockingModel as BioDocking1,
        DockingPoseModelV2 as BioDocking2,
    )
    assert BioDocking1 is UniMolDockingModel
    assert BioDocking2 is DockingPoseModelV2


def test_chemistry_api():
    # Via k3_node.models.chemistry
    from k3_node.models.chemistry import (
        AttentiveFP,
        DimeNet,
        DimeNetPlusPlus,
        GROVER,
        MoleBERT,
        NeuralFingerprint,
        SchNet,
        ViSNet,
        GNNFF,
        Graphormer,
        Graphormer3D,
        UniMolModel,
        UniMolConfGenModel,
        UniMol2Model,
        UniMolPlusPCQModel,
        UniMolPlusOC20Model,
    )
    assert AttentiveFP is not None
    assert DimeNet is not None
    assert SchNet is not None
    assert UniMolModel is not None

    # Via top-level k3_node.chemistry
    from k3_node.chemistry import (
        SchNet as ChemSchNet,
        DimeNet as ChemDimeNet,
        UniMolModel as ChemUniMol,
    )
    assert ChemSchNet is SchNet
    assert ChemDimeNet is DimeNet
    assert ChemUniMol is UniMolModel


def test_backward_compatibility():
    # All models still exportable from top-level k3_node.models
    from k3_node.models import (
        MEGNet,
        M3GNet,
        TensorNet,
        CHGNet,
        SO3Net,
        GRACE,
        QET,
        SchNet,
        DimeNet,
        UniMolModel,
        UniMolDockingModel,
        DockingPoseModelV2,
    )
    assert MEGNet is not None
    assert SchNet is not None
    assert UniMolModel is not None

