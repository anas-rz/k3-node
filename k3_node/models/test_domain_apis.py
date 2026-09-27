"""Test domain-specific API organization (applications: materials, bio, chemistry)."""

import pytest


def test_clean_main_api():
    """Ensure main k3_node API is clean and domain packages are under applications."""
    import k3_node

    assert not hasattr(k3_node, "materials"), "k3_node should not expose materials directly"
    assert not hasattr(k3_node, "bio"), "k3_node should not expose bio directly"
    assert not hasattr(k3_node, "chemistry"), "k3_node should not expose chemistry directly"

    assert hasattr(k3_node, "applications"), "k3_node must expose applications"
    assert hasattr(k3_node.applications, "materials"), "applications must expose materials"
    assert hasattr(k3_node.applications, "bio"), "applications must expose bio"
    assert hasattr(k3_node.applications, "chemistry"), "applications must expose chemistry"


def test_materials_application_api():
    # Via k3_node.applications.materials
    from k3_node.applications.materials import (
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
        basis,
        core,
        readout,
    )
    assert MEGNet is not None
    assert M3GNet is not None
    assert TensorNet is not None
    assert CHGNet is not None
    assert SO3Net is not None
    assert GRACE is not None
    assert QET is not None
    assert basis is not None
    assert core is not None
    assert readout is not None

    # Via k3_node.models.materials (backward compatibility)
    from k3_node.models.materials import (
        MEGNet as MatMEGNet,
        M3GNet as MatM3GNet,
        TensorNet as MatTensorNet,
    )
    assert MatMEGNet is MEGNet
    assert MatM3GNet is M3GNet
    assert MatTensorNet is TensorNet


def test_bio_application_api():
    # Via k3_node.applications.bio
    from k3_node.applications.bio import (
        UniMolDockingModel,
        DockingPoseModelV2,
        download_unimol_checkpoint,
        load_unimol_weights,
        download_unimol_docking_checkpoint,
        load_unimol_docking_weights,
    )
    assert UniMolDockingModel is not None
    assert DockingPoseModelV2 is not None

    # Via k3_node.models.bio (backward compatibility)
    from k3_node.models.bio import (
        UniMolDockingModel as BioDocking1,
        DockingPoseModelV2 as BioDocking2,
    )
    assert BioDocking1 is UniMolDockingModel
    assert BioDocking2 is DockingPoseModelV2


def test_chemistry_application_api():
    # Via k3_node.applications.chemistry
    from k3_node.applications.chemistry import (
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

    # Via k3_node.models.chemistry (backward compatibility)
    from k3_node.models.chemistry import (
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
