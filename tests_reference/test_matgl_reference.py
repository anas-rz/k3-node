"""
Reference parity tests against MatGL (Materials Graph Library) implementation.

These tests verify that k3-node's multi-backend ports of materials layers, functions,
and blocks produce numerically equivalent outputs to their MatGL / PyTorch counterparts.
"""

from __future__ import annotations

import os
import sys
import types
import os.path as osp
import numpy as np
import pytest
import torch
import keras
from keras import ops

# Dynamic mock loader for heavy dependencies needed by matgl import
class AutoMockFinder:
    def find_spec(self, fullname, path, target=None):
        if any(fullname.startswith(p) for p in ['pymatgen', 'torch_geometric', 'dgl', 'lightning', 'torchmetrics', 'monty', 'ase']):
            import importlib.machinery
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            spec.loader = AutoMockLoader()
            return spec
        return None

class AutoMockLoader:
    def create_module(self, spec):
        m = types.ModuleType(spec.name)
        m.__path__ = []
        return m
    def exec_module(self, module):
        if module.__name__ == 'pymatgen.core.periodic_table':
            class DummyEl:
                def __init__(self, s, z): self.symbol = s; self.Z = z
            module.Element = [DummyEl('H', 1), DummyEl('C', 6), DummyEl('O', 8)]
        elif module.__name__ == 'torch_geometric.nn':
            class DummyMP(torch.nn.Module):
                def __init__(self, *args, **kwargs): super().__init__()
            module.MessagePassing = DummyMP
            module.global_add_pool = lambda x, batch: torch.zeros(1)
            module.global_max_pool = lambda x, batch: torch.zeros(1)
            module.global_mean_pool = lambda x, batch: torch.zeros(1)
        elif module.__name__ == 'torch_geometric.nn.aggr':
            class DummySet2Set(torch.nn.Module):
                def __init__(self, *args, **kwargs): super().__init__()
            module.Set2Set = DummySet2Set
        elif module.__name__ == 'ase.data':
            module.atomic_numbers = {'H': 1, 'C': 6, 'O': 8}
        elif module.__name__ == 'lightning':
            class DummyLM(torch.nn.Module):
                def __init__(self, *args, **kwargs): super().__init__()
            module.LightningModule = DummyLM

if not any(isinstance(finder, AutoMockFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, AutoMockFinder())

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
MATGL_SRC = osp.join(REPO_ROOT, "matgl", "src")
if MATGL_SRC not in sys.path:
    sys.path.insert(0, MATGL_SRC)

# MatGL reference imports
from matgl.layers import BondExpansion as RefBondExpansion
from matgl.layers import RadialBesselFunction as RefRBF
from matgl.layers._activations import SoftPlus2 as RefSoftPlus2, SoftExponential as RefSoftExponential
from matgl.layers._so3 import RealSphericalHarmonics as RefRSH
from matgl.layers import MLP as RefMLP, GatedMLP as RefGatedMLP
from matgl.utils.maths import vector_to_skewtensor as ref_v2skew, vector_to_symtensor as ref_v2sym, decompose_tensor as ref_decomp

# K3 imports
from k3_node.models.materials.basis import BondExpansion as K3BondExpansion, RadialBesselFunction as K3RBF
from k3_node.models.materials.core import SoftPlus2 as K3SoftPlus2, SoftExponential as K3SoftExponential
from k3_node.models.materials.so3net import RealSphericalHarmonics as K3RSH
from k3_node.models.materials.core import (
    vector_to_skewtensor as k3_v2skew,
    vector_to_symtensor as k3_v2sym,
    decompose_tensor as k3_decomp,
    MLP as K3MLP,
    GatedMLP as K3GatedMLP,
    EmbeddingBlock as K3EmbeddingBlock,
)


def test_reference_softplus2():
    x_np = np.array([-3.0, -1.0, 0.0, 0.5, 2.0, 5.0], dtype=np.float32)
    ref_sp2 = RefSoftPlus2()
    k3_sp2 = K3SoftPlus2()

    out_ref = ref_sp2(torch.from_numpy(x_np)).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_sp2(x_np))
    np.testing.assert_allclose(out_k3, out_ref, rtol=1e-5, atol=1e-5)


def test_reference_soft_exponential():
    x_np = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    for alpha in [-0.5, 0.0, 0.5]:
        ref_se = RefSoftExponential(alpha=alpha)
        k3_se = K3SoftExponential(alpha=alpha)

        out_ref = ref_se(torch.from_numpy(x_np)).detach().numpy()
        out_k3 = ops.convert_to_numpy(k3_se(x_np))
        np.testing.assert_allclose(out_k3, out_ref, rtol=1e-5, atol=1e-5)


def test_reference_vector_to_skewtensor():
    v_np = np.random.randn(8, 3).astype(np.float32)
    out_ref = ref_v2skew(torch.from_numpy(v_np)).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_v2skew(v_np))
    np.testing.assert_allclose(out_k3, out_ref, rtol=1e-5, atol=1e-5)


def test_reference_vector_to_symtensor():
    v_np = np.random.randn(8, 3).astype(np.float32)
    out_ref = ref_v2sym(torch.from_numpy(v_np)).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_v2sym(v_np))
    np.testing.assert_allclose(out_k3, out_ref, rtol=1e-5, atol=1e-5)


def test_reference_decompose_tensor():
    mat_np = np.random.randn(6, 3, 3).astype(np.float32)
    ref_d, ref_s, ref_a = ref_decomp(torch.from_numpy(mat_np))
    k3_d, k3_s, k3_a = k3_decomp(mat_np)

    np.testing.assert_allclose(ops.convert_to_numpy(k3_d), ref_d.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ops.convert_to_numpy(k3_s), ref_s.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ops.convert_to_numpy(k3_a), ref_a.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_reference_radial_bessel_function():
    r_np = np.linspace(0.5, 4.8, 16).astype(np.float32)
    ref_rbf = RefRBF(max_n=16, cutoff=5.0)
    k3_rbf = K3RBF(max_n=16, cutoff=5.0)

    out_ref = ref_rbf(torch.from_numpy(r_np)).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_rbf(r_np))
    np.testing.assert_allclose(out_k3, out_ref, rtol=1e-5, atol=1e-5)


def test_reference_bond_expansion_gaussian():
    r_np = np.linspace(0.5, 4.5, 12).astype(np.float32)
    ref_be = RefBondExpansion(max_l=16, max_n=16, cutoff=5.0, rbf_type="Gaussian")
    k3_be = K3BondExpansion(max_l=16, max_n=16, cutoff=5.0, rbf_type="Gaussian")

    out_ref = ref_be(torch.from_numpy(r_np)).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_be(r_np))
    np.testing.assert_allclose(out_k3, out_ref, rtol=1e-5, atol=1e-5)


def test_reference_real_spherical_harmonics():
    v = np.random.randn(8, 3).astype(np.float32)
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)

    ref_rsh = RefRSH(lmax=2)
    k3_rsh = K3RSH(lmax=2)

    out_ref = ref_rsh(torch.from_numpy(v)).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_rsh(v))
    np.testing.assert_allclose(out_k3, out_ref, rtol=1e-5, atol=1e-5)


def test_reference_mlp():
    ref_mlp = RefMLP(dims=[16, 32, 16], activation=torch.nn.SiLU(), activate_last=True)
    k3_mlp = K3MLP(dims=[16, 32, 16], activation="silu", activate_last=True, use_bias=True)

    x_np = np.random.randn(4, 16).astype(np.float32)
    _ = k3_mlp(x_np)

    k3_mlp.dense_layers[0].kernel.assign(ref_mlp.layers[0].weight.detach().numpy().T)
    k3_mlp.dense_layers[0].bias.assign(ref_mlp.layers[0].bias.detach().numpy())
    k3_mlp.dense_layers[1].kernel.assign(ref_mlp.layers[2].weight.detach().numpy().T)
    k3_mlp.dense_layers[1].bias.assign(ref_mlp.layers[2].bias.detach().numpy())

    out_ref = ref_mlp(torch.from_numpy(x_np)).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_mlp(x_np))
    np.testing.assert_allclose(out_k3, out_ref, rtol=1e-5, atol=1e-5)


def test_reference_gated_mlp():
    ref_gmlp = RefGatedMLP(in_feats=16, dims=[32, 16], activate_last=False, use_bias=True)
    k3_gmlp = K3GatedMLP(in_feats=16, dims=[32, 16], activate_last=False, use_bias=True)

    x_np = np.random.randn(4, 16).astype(np.float32)
    _ = k3_gmlp(x_np)

    k3_gmlp.val_layers[0].kernel.assign(ref_gmlp.layers[0].weight.detach().numpy().T)
    k3_gmlp.val_layers[0].bias.assign(ref_gmlp.layers[0].bias.detach().numpy())
    k3_gmlp.val_layers[1].kernel.assign(ref_gmlp.layers[2].weight.detach().numpy().T)
    k3_gmlp.val_layers[1].bias.assign(ref_gmlp.layers[2].bias.detach().numpy())

    k3_gmlp.gate_layers[0].kernel.assign(ref_gmlp.gates[0].weight.detach().numpy().T)
    k3_gmlp.gate_layers[0].bias.assign(ref_gmlp.gates[0].bias.detach().numpy())
    k3_gmlp.gate_layers[1].kernel.assign(ref_gmlp.gates[2].weight.detach().numpy().T)
    k3_gmlp.gate_layers[1].bias.assign(ref_gmlp.gates[2].bias.detach().numpy())

    out_ref = ref_gmlp(torch.from_numpy(x_np)).detach().numpy()
    out_k3 = ops.convert_to_numpy(k3_gmlp(x_np))
    np.testing.assert_allclose(out_k3, out_ref, rtol=1e-5, atol=1e-5)

