"""
Sanity-check reference parity tests against PyTorch Geometric.

These tests verify that k3-node's ports of `torch_geometric.nn.models`
produce numerically equivalent outputs to their PyG counterparts. These
tests are intended for local validation and documentation, and are not run
as part of the default GitHub Actions test suites.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import torch
import keras
from keras import ops

from torch_geometric.nn.models import MLP as PyGMLP
from torch_geometric.nn.models import ARLinkPredictor as PyGARLinkPredictor
from torch_geometric.nn.models import GAE as PyGGAE
from torch_geometric.nn.models import DeepGraphInfomax as PyGDeepGraphInfomax
from torch_geometric.nn import DeepGCNLayer as PyGDeepGCNLayer
from torch_geometric.nn import GCNConv as PyGGCNConv
from torch_geometric.nn import LayerNorm as PyGLayerNorm
from torch_geometric.nn.models.attentive_fp import GATEConv as PyGGATEConv
from torch_geometric.nn.models.attentive_fp import AttentiveFP as PyGAttentiveFP

from k3_node.models import MLP, ARLinkPredictor, GAE, DeepGraphInfomax, DeepGCNLayer, AttentiveFP
from k3_node.models.attentive_fp import GATEConv
from k3_node.layers.conv import GCNConv
from k3_node.layers.norm import LayerNorm


def _copy_gru(pyg_cell, k3_cell, units):
    def reorder(w):
        r, z, n = w[:units], w[units:2 * units], w[2 * units:3 * units]
        return torch.cat([z, r, n], dim=0)

    kernel = reorder(pyg_cell.weight_ih).t().contiguous()
    recurrent_kernel = reorder(pyg_cell.weight_hh).t().contiguous()
    bias_ih = reorder(pyg_cell.bias_ih.unsqueeze(-1)).squeeze(-1)
    bias_hh = reorder(pyg_cell.bias_hh.unsqueeze(-1)).squeeze(-1)
    bias = torch.stack([bias_ih, bias_hh], dim=0)

    k3_cell.kernel.assign(kernel.detach())
    k3_cell.recurrent_kernel.assign(recurrent_kernel.detach())
    k3_cell.bias.assign(bias.detach())


def _copy_gat(pyg_conv, k3_conv):
    k3_conv.att_src.assign(pyg_conv.att_src.detach())
    k3_conv.att_dst.assign(pyg_conv.att_dst.detach())
    k3_conv.bias.assign(pyg_conv.bias.detach())
    k3_conv.lin.kernel.assign(pyg_conv.lin.weight.detach().t())


def test_reference_mlp():
    torch.manual_seed(42)
    pyg = PyGMLP([16, 32, 32, 8], norm=None, plain_last=True)
    k3 = MLP([16, 32, 32, 8], norm=None, plain_last=True)
    for pl, kl in zip(pyg.lins, k3.lins):
        kl.kernel.assign(pl.weight.detach().t())
        kl.bias.assign(pl.bias.detach())

    x = torch.randn(4, 16)
    out_pyg = pyg(x)
    out_k3 = k3(x)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-5


def test_reference_ar_link_predictor():
    torch.manual_seed(42)
    pyg = PyGARLinkPredictor(in_channels=16, hidden_channels=32, num_layers=2)
    k3 = ARLinkPredictor(in_channels=16, hidden_channels=32, num_layers=2)
    for pl, kl in zip(pyg.lins, k3.lins):
        kl.kernel.assign(pl.weight.detach().t())
        kl.bias.assign(pl.bias.detach())
    k3.lin_attract.kernel.assign(pyg.lin_attract.weight.detach().t())
    k3.lin_attract.bias.assign(pyg.lin_attract.bias.detach())
    k3.lin_repel.kernel.assign(pyg.lin_repel.weight.detach().t())
    k3.lin_repel.bias.assign(pyg.lin_repel.bias.detach())

    x = torch.randn(6, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])

    out_pyg = pyg(x, edge_index)
    out_k3 = k3(x, edge_index)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-5


def test_reference_gae():
    # `encoder=lambda x: x` makes this an exact (weight-free) check.
    pyg = PyGGAE(encoder=lambda x: x)
    k3 = GAE(encoder=lambda x: x)

    x = torch.tensor([[1.0, -1.0], [1.0, 2.0], [2.0, 1.0]])
    z_pyg = pyg.encode(x)
    z_k3 = k3.encode(x)
    assert np.allclose(z_pyg.numpy(), ops.convert_to_numpy(z_k3), atol=1e-6)

    adj_pyg = pyg.decoder.forward_all(z_pyg)
    adj_k3 = k3.decoder.forward_all(z_k3)
    assert np.allclose(adj_pyg.detach().numpy(), ops.convert_to_numpy(adj_k3), atol=1e-6)

    edge_index = torch.tensor([[0, 1], [1, 2]])
    val_pyg = pyg.decode(z_pyg, edge_index)
    val_k3 = k3.decode(z_k3, edge_index)
    assert np.allclose(val_pyg.detach().numpy(), ops.convert_to_numpy(val_k3), atol=1e-6)


def test_reference_deep_graph_infomax():
    torch.manual_seed(42)
    pyg = PyGDeepGraphInfomax(
        hidden_channels=16,
        encoder=lambda x: x,
        summary=lambda z, *args: z.mean(dim=0),
        corruption=lambda x: x + 1,
    )
    k3 = DeepGraphInfomax(
        hidden_channels=16,
        encoder=lambda x: x,
        summary=lambda z, *args: ops.mean(z, axis=0),
        corruption=lambda x: x + 1,
    )
    k3.weight.assign(pyg.weight.detach())

    x = torch.randn(10, 16)
    pos_z_pyg, neg_z_pyg, summary_pyg = pyg(x)
    pos_z_k3, neg_z_k3, summary_k3 = k3(x)

    assert np.allclose(pos_z_pyg.numpy(), ops.convert_to_numpy(pos_z_k3), atol=1e-6)
    assert np.allclose(neg_z_pyg.numpy(), ops.convert_to_numpy(neg_z_k3), atol=1e-6)
    assert np.allclose(summary_pyg.numpy(), ops.convert_to_numpy(summary_k3), atol=1e-6)

    loss_pyg = pyg.loss(pos_z_pyg, neg_z_pyg, summary_pyg)
    loss_k3 = k3.loss(pos_z_k3, neg_z_k3, summary_k3)
    assert abs(loss_pyg.item() - float(ops.convert_to_numpy(loss_k3))) < 1e-5


def test_reference_deepgcn_layer():
    torch.manual_seed(42)
    conv_pyg = PyGGCNConv(8, 8)
    norm_pyg = PyGLayerNorm(8)
    layer_pyg = PyGDeepGCNLayer(conv_pyg, norm_pyg, torch.nn.ReLU(), block="res+")

    conv_k3 = GCNConv(8, 8)
    conv_k3.build((None, 8))
    conv_k3.lin.kernel.assign(conv_pyg.lin.weight.detach().t())
    conv_k3.bias.assign(conv_pyg.bias.detach())
    norm_k3 = LayerNorm(8)
    norm_k3.weight.assign(norm_pyg.weight.detach())
    norm_k3.bias.assign(norm_pyg.bias.detach())
    layer_k3 = DeepGCNLayer(conv_k3, norm_k3, keras.layers.ReLU(), block="res+")

    x = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]])

    layer_pyg.eval()
    out_pyg = layer_pyg(x, edge_index)
    out_k3 = layer_k3(x, edge_index, training=False)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-4


def test_reference_gate_conv():
    torch.manual_seed(0)
    pyg = PyGGATEConv(16, 16, edge_dim=3, dropout=0.0)
    k3 = GATEConv(16, 16, edge_dim=3, dropout=0.0)

    k3.att_l.assign(pyg.att_l.detach())
    k3.att_r.assign(pyg.att_r.detach())
    k3.bias.assign(pyg.bias.detach())
    k3.lin1.kernel.assign(pyg.lin1.weight.detach().t())
    k3.lin2.kernel.assign(pyg.lin2.weight.detach().t())

    x = torch.randn(4, 16)
    edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]])
    edge_attr = torch.randn(edge_index.size(1), 3)

    out_pyg = pyg(x, edge_index, edge_attr)
    out_k3 = k3(x, edge_index, edge_attr)
    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-5


def test_reference_attentive_fp():
    torch.manual_seed(0)
    hidden = 16
    pyg = PyGAttentiveFP(8, hidden, 4, edge_dim=3, num_layers=2, num_timesteps=2)
    k3 = AttentiveFP(8, hidden, 4, edge_dim=3, num_layers=2, num_timesteps=2)

    k3.lin1.kernel.assign(pyg.lin1.weight.detach().t())
    k3.lin1.bias.assign(pyg.lin1.bias.detach())

    k3.gate_conv.att_l.assign(pyg.gate_conv.att_l.detach())
    k3.gate_conv.att_r.assign(pyg.gate_conv.att_r.detach())
    k3.gate_conv.bias.assign(pyg.gate_conv.bias.detach())
    k3.gate_conv.lin1.kernel.assign(pyg.gate_conv.lin1.weight.detach().t())
    k3.gate_conv.lin2.kernel.assign(pyg.gate_conv.lin2.weight.detach().t())
    _copy_gru(pyg.gru, k3.gru, hidden)

    for conv_pyg, conv_k3 in zip(pyg.atom_convs, k3.atom_convs):
        _copy_gat(conv_pyg, conv_k3)
    for gru_pyg, gru_k3 in zip(pyg.atom_grus, k3.atom_grus):
        _copy_gru(gru_pyg, gru_k3, hidden)

    _copy_gat(pyg.mol_conv, k3.mol_conv)
    _copy_gru(pyg.mol_gru, k3.mol_gru, hidden)

    k3.lin2.kernel.assign(pyg.lin2.weight.detach().t())
    k3.lin2.bias.assign(pyg.lin2.bias.detach())

    x = torch.randn(6, 8)
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 0, 4, 3]])
    edge_attr = torch.randn(edge_index.size(1), 3)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    pyg.eval()
    out_pyg = pyg(x, edge_index, edge_attr, batch)
    out_k3 = k3(x, edge_index, edge_attr, batch, training=False)

    diff = (out_pyg - out_k3).abs().max().item()
    assert diff < 1e-3
