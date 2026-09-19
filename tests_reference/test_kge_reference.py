"""
Sanity-check reference parity tests against PyTorch Geometric.

These tests verify that k3-node KGE (knowledge graph embedding) models
produce numerically equivalent outputs to torch_geometric.nn.kge reference
implementations. These tests are intended for local validation and
documentation, and are not run as part of the default GitHub Actions test
suites.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import torch

from torch_geometric.nn.kge import (
    TransE as PyGTransE,
    DistMult as PyGDistMult,
    ComplEx as PyGComplEx,
    RotatE as PyGRotatE,
)
from k3_node.layers.kge import TransE, DistMult, ComplEx, RotatE


def _copy_embedding(pyg_emb, k3_emb):
    k3_emb.embeddings.assign(pyg_emb.weight.detach())


def _triplet():
    head_index = torch.tensor([0, 2, 4, 6, 8])
    rel_type = torch.tensor([0, 1, 2, 3, 4])
    tail_index = torch.tensor([1, 3, 5, 7, 9])
    return head_index, rel_type, tail_index


def test_reference_transe():
    torch.manual_seed(42)
    pyg_model = PyGTransE(num_nodes=10, num_relations=5, hidden_channels=16)
    k3_model = TransE(num_nodes=10, num_relations=5, hidden_channels=16)
    _copy_embedding(pyg_model.node_emb, k3_model.node_emb)
    _copy_embedding(pyg_model.rel_emb, k3_model.rel_emb)

    head_index, rel_type, tail_index = _triplet()
    out_pyg = pyg_model(head_index, rel_type, tail_index)
    out_k3 = k3_model(head_index, rel_type, tail_index)

    assert np.allclose(out_pyg.detach().numpy(), out_k3.detach().numpy(), atol=1e-5)


def test_reference_distmult():
    torch.manual_seed(42)
    pyg_model = PyGDistMult(num_nodes=10, num_relations=5, hidden_channels=16)
    k3_model = DistMult(num_nodes=10, num_relations=5, hidden_channels=16)
    _copy_embedding(pyg_model.node_emb, k3_model.node_emb)
    _copy_embedding(pyg_model.rel_emb, k3_model.rel_emb)

    head_index, rel_type, tail_index = _triplet()
    out_pyg = pyg_model(head_index, rel_type, tail_index)
    out_k3 = k3_model(head_index, rel_type, tail_index)

    assert np.allclose(out_pyg.detach().numpy(), out_k3.detach().numpy(), atol=1e-5)


def test_reference_complex():
    torch.manual_seed(42)
    pyg_model = PyGComplEx(num_nodes=10, num_relations=5, hidden_channels=16)
    k3_model = ComplEx(num_nodes=10, num_relations=5, hidden_channels=16)
    _copy_embedding(pyg_model.node_emb, k3_model.node_emb)
    _copy_embedding(pyg_model.node_emb_im, k3_model.node_emb_im)
    _copy_embedding(pyg_model.rel_emb, k3_model.rel_emb)
    _copy_embedding(pyg_model.rel_emb_im, k3_model.rel_emb_im)

    head_index, rel_type, tail_index = _triplet()
    out_pyg = pyg_model(head_index, rel_type, tail_index)
    out_k3 = k3_model(head_index, rel_type, tail_index)

    assert np.allclose(out_pyg.detach().numpy(), out_k3.detach().numpy(), atol=1e-5)


def test_reference_rotate():
    torch.manual_seed(42)
    pyg_model = PyGRotatE(num_nodes=10, num_relations=5, hidden_channels=16)
    k3_model = RotatE(num_nodes=10, num_relations=5, hidden_channels=16)
    _copy_embedding(pyg_model.node_emb, k3_model.node_emb)
    _copy_embedding(pyg_model.node_emb_im, k3_model.node_emb_im)
    _copy_embedding(pyg_model.rel_emb, k3_model.rel_emb)

    head_index, rel_type, tail_index = _triplet()
    out_pyg = pyg_model(head_index, rel_type, tail_index)
    out_k3 = k3_model(head_index, rel_type, tail_index)

    assert np.allclose(out_pyg.detach().numpy(), out_k3.detach().numpy(), atol=1e-5)


def test_reference_complex_scoring_fixed_case():
    # Matches torch_geometric/test/nn/kge/test_complex.py::test_complex_scoring.
    pyg_model = PyGComplEx(num_nodes=5, num_relations=2, hidden_channels=1)
    pyg_model.node_emb.weight.data = torch.tensor([[2.], [3.], [5.], [1.], [2.]])
    pyg_model.node_emb_im.weight.data = torch.tensor([[4.], [1.], [3.], [1.], [2.]])
    pyg_model.rel_emb.weight.data = torch.tensor([[2.], [3.]])
    pyg_model.rel_emb_im.weight.data = torch.tensor([[3.], [1.]])

    k3_model = ComplEx(num_nodes=5, num_relations=2, hidden_channels=1)
    _copy_embedding(pyg_model.node_emb, k3_model.node_emb)
    _copy_embedding(pyg_model.node_emb_im, k3_model.node_emb_im)
    _copy_embedding(pyg_model.rel_emb, k3_model.rel_emb)
    _copy_embedding(pyg_model.rel_emb_im, k3_model.rel_emb_im)

    head_index = torch.tensor([1, 3])
    rel_type = torch.tensor([1, 0])
    tail_index = torch.tensor([2, 4])

    out_pyg = pyg_model(head_index, rel_type, tail_index)
    out_k3 = k3_model(head_index, rel_type, tail_index)

    assert out_pyg.tolist() == [58.0, 8.0]
    assert np.allclose(out_pyg.detach().numpy(), out_k3.detach().numpy(), atol=1e-5)
